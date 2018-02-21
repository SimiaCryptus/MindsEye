/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import jcuda.jcudnn.cudnnTensorDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgTileSelectLayer extends LayerBase implements MultiPrecision<ImgTileSelectLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileSelectLayer.class);
  private int positionX;
  private int positionY;
  
  private int width;
  private int height;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgTileSelectLayer() {
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param width  the size x
   * @param height the size y
   */
  public ImgTileSelectLayer(int width, int height, final int positionX, final int positionY) {
    this.width = width;
    this.height = height;
    this.positionX = positionX;
    this.positionY = positionY;
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileSelectLayer(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    width = json.get("width").getAsInt();
    height = json.get("height").getAsInt();
    positionX = json.get("positionX").getAsInt();
    positionY = json.get("positionY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgTileSelectLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer.class);
  }
  
  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    assert 1 == inObj.length;
    final Result in = inObj[0];
    assert 3 == in.getData().getDimensions().length;
    final int length = in.getData().length();
    @Nonnull int[] dimIn = in.getData().getDimensions();
    if (dimIn[0] == width && dimIn[1] == height) {
      in.addRef();
      in.getData().addRef();
      return in;
    }
    @Nonnull final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = width;
    dimOut[1] = height;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final TensorList outputData = CudaSystem.eval(gpu -> {
      @Nullable final CudaPtr inputBuffer = CudaPtr.getCudaPtr(precision, in.getData());
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      @Nonnull final CudaPtr outputBuffer = CudaPtr.allocate(gpu.getDeviceNumber(), (long) length * dimOut[2] * dimOut[1] * dimOut[0] * precision.size, MemoryType.Managed, dirty);
      copy(gpu, length, dimIn, inputBuffer, dimOut, outputBuffer, this.positionX, this.positionY);
      gpu.registerForCleanup(inputBuffer);
      return CudaTensorList.wrap(outputBuffer, length, dimOut, precision);
    });
    return new Result(outputData, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList error) -> {
      if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
        throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
      }
      if (error.length() != outputData.length()) {
        throw new AssertionError(error.length() + " != " + outputData.length());
      }
      assert error.length() == in.getData().length();
      if (in.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
          @Nullable final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          @Nonnull final CudaPtr passbackBuffer = CudaPtr.allocate(gpu.getDeviceNumber(), (long) (length * dimIn[2] * dimIn[1] * dimIn[0] * precision.size), MemoryType.Managed, dirty);
          copy(gpu, length, dimOut, errorPtr, dimIn, passbackBuffer, -this.positionX, -this.positionY);
          gpu.registerForCleanup(errorPtr);
          return CudaTensorList.wrap(passbackBuffer, length, dimIn, precision);
        });
        in.accumulate(buffer, passbackTensorList);
        passbackTensorList.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  /**
   * Copy.
   *
   * @param gpu                   the gpu
   * @param length                the length
   * @param sourceDimensions      the dim in
   * @param source                the input buffer
   * @param destinationDimensions the dim out
   * @param destination           the output buffer
   */
  public int[] copy(@Nonnull CudnnHandle gpu, int length, @Nonnull int[] sourceDimensions, @Nonnull CudaPtr source, @Nonnull int[] destinationDimensions, @Nonnull CudaPtr destination, int positionX, int positionY) {
    if (3 != sourceDimensions.length) throw new IllegalArgumentException("inputDimensions.length");
    if (3 != destinationDimensions.length) throw new IllegalArgumentException("dimOut.length");
    int bands = sourceDimensions[2];
    if (bands != destinationDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, destinationDimensions[2]));
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(sourceDimensions, destinationDimensions, new int[]{positionX, positionY, 0});
    @Nonnull final CudaResource<cudnnTensorDescriptor> sourceViewDescriptor = CudaSystem.newTensorDescriptor(
      precision.code,//
      length,//
      viewDim[2],//
      viewDim[1],//
      viewDim[0],//
      bands * sourceDimensions[1] * sourceDimensions[0],//
      sourceDimensions[1] * sourceDimensions[0],//
      sourceDimensions[0],//
      1);
    @Nonnull final CudaResource<cudnnTensorDescriptor> destinationViewDescriptor = CudaSystem.newTensorDescriptor(
      precision.code,//
      length,//
      viewDim[2],//
      viewDim[1],//
      viewDim[0],//
      destinationDimensions[2] * destinationDimensions[1] * destinationDimensions[0],//
      destinationDimensions[1] * destinationDimensions[0],//
      destinationDimensions[0],//
      1);
    int sourceOffset = 0;
    int destinationOffset = 0;
    
    if (positionX > 0) {
      destinationOffset += Math.abs(positionX) / 2;
    }
    else {
      sourceOffset += Math.abs(positionX) / 2;
    }
    if (positionY > 0) {
      destinationOffset += destinationDimensions[0] * Math.abs((positionY) / 2);
    }
    else {
      sourceOffset += sourceDimensions[0] * (Math.abs(positionY) / 2);
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.dim(viewDim) <= Tensor.dim(sourceDimensions);
    assert destinationOffset + Tensor.dim(viewDim) <= Tensor.dim(destinationDimensions);
    
    CudaSystem.handle(gpu.cudnnTransformTensor(
      precision.getPointer(1.0),
      sourceViewDescriptor.getPtr(), source.getPtr().withByteOffset(sourceOffset * precision.size),
      precision.getPointer(1.0),
      destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(destinationOffset * precision.size)
    ));
    gpu.registerForCleanup(sourceViewDescriptor, destinationViewDescriptor);
    return viewDim;
    
  }
  
  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @return the int [ ]
   */
  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i ->
      Math.min(sourceDimensions[i] + offset[i], destinationDimensions[i]) -
        Math.max(offset[i], 0)
    );
    return viewDim;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("height", height);
    json.addProperty("width", width);
    json.addProperty("positionX", positionX);
    json.addProperty("positionY", positionY);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgTileSelectLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
