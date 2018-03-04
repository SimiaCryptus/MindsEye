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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgTileSelectLayer extends LayerBase implements MultiPrecision<ImgTileSelectLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileSelectLayer.class);
  private int positionX;
  private int positionY;
  
  private int sizeY;
  private int sizeX;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgTileSelectLayer() {
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX     the size y
   * @param sizeY     the size x
   * @param positionX the position x
   * @param positionY the position y
   */
  public ImgTileSelectLayer(int sizeX, int sizeY, final int positionX, final int positionY) {
    this.sizeY = sizeY;
    this.sizeX = sizeX;
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
    sizeY = json.get("sizeX").getAsInt();
    sizeX = json.get("sizeY").getAsInt();
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
    final Result input = inObj[0];
    assert 3 == input.getData().getDimensions().length;
    final int length = input.getData().length();
    @Nonnull int[] dimIn = input.getData().getDimensions();
    if (dimIn[0] == sizeY && dimIn[1] == sizeX) {
      input.addRef();
      input.getData().addRef();
      return input;
    }
    @Nonnull final int[] dimOut = getViewDimensions(dimIn, new int[]{sizeY, sizeX, dimIn[2]}, new int[]{positionX, positionY, 0});
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final TensorList outputData = CudaSystem.eval(gpu -> {
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      boolean dirty = dimOut[0] == dimIn[0] && dimOut[1] == dimIn[1];
      CudaTensor cudaTensor = copy(gpu, input.getData(), length, dimIn, dimOut, dirty, this.positionX, this.positionY);
      return CudaTensorList.wrap(cudaTensor, length, dimOut, precision);
    }, input.getData());
    return new Result(outputData, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList error) -> {
      if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
        throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
      }
      if (error.length() != outputData.length()) {
        throw new AssertionError(error.length() + " != " + outputData.length());
      }
      assert error.length() == input.getData().length();
      if (input.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          CudaTensor cudaTensor = copy(gpu, error, length, dimOut, dimIn, dirty, -this.positionX, -this.positionY);
          return CudaTensorList.wrap(cudaTensor, length, dimIn, precision);
        }, error);
        input.accumulate(buffer, passbackTensorList);
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
  
  public CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int length, final int[] inputDimensions, final int[] outputDimensions, final boolean dirty, final int positionX, final int positionY) {
    if (3 != inputDimensions.length) throw new IllegalArgumentException("inputDimensions.length");
    if (3 != outputDimensions.length) throw new IllegalArgumentException("dimOut.length");
    int bands = inputDimensions[2];
    if (bands != outputDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, outputDimensions[2]));
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(inputDimensions, outputDimensions, new int[]{positionX, positionY, 0});
    @Nullable final CudaTensor inputTensor = gpu.getTensor(input, precision, MemoryType.Device);
    
    int sourceOffset = 0;
    int destinationOffset = 0;
    if (positionX < 0) {
      destinationOffset += Math.abs(positionX);
    }
    else {
      sourceOffset += Math.abs(positionX);
    }
    if (positionY < 0) {
      destinationOffset += outputDimensions[0] * Math.abs((positionY));
    }
    else {
      sourceOffset += inputTensor.descriptor.hStride * (Math.abs(positionY));
    }
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= Tensor.length(inputDimensions);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(outputDimensions);
  
    @Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(
      precision,//
      length,//
      viewDim[2],//
      viewDim[1],//
      viewDim[0],//
      inputTensor.descriptor.nStride,//
      inputTensor.descriptor.cStride,//
      inputTensor.descriptor.hStride,//
      inputTensor.descriptor.wStride);
    if (Arrays.equals(viewDim, outputDimensions)) {
      assert sourceOffset >= 0;
      assert destinationOffset == 0;
      return CudaTensor.wrap(inputTensor.memory.withByteOffset(sourceOffset * precision.size), sourceViewDescriptor, precision);
    }
    
    @Nonnull final CudaMemory outputPtr = gpu.allocate((long) length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size, MemoryType.Managed, dirty);
    @Nonnull final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(
      precision,//
      length,//
      viewDim[2],//
      viewDim[1],//
      viewDim[0],//
      outputDimensions[2] * outputDimensions[1] * outputDimensions[0],//
      outputDimensions[1] * outputDimensions[0],//
      outputDimensions[0],//
      1);
    CudaSystem.handle(gpu.cudnnTransformTensor(
      precision.getPointer(1.0),
      sourceViewDescriptor.getPtr(), inputTensor.memory.getPtr().withByteOffset(sourceOffset * precision.size),
      precision.getPointer(1.0),
      destinationViewDescriptor.getPtr(), outputPtr.getPtr().withByteOffset(destinationOffset * precision.size)
    ));
    Arrays.stream(new ReferenceCounting[]{sourceViewDescriptor}).forEach(ReferenceCounting::freeRef);
    
    @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
      precision,//
      length,//
      outputDimensions[2],//
      outputDimensions[1],//
      outputDimensions[0],//
      outputDimensions[2] * outputDimensions[1] * outputDimensions[0],//
      outputDimensions[1] * outputDimensions[0],//
      outputDimensions[0],//
      1);
    Stream.<ReferenceCounting>of(inputTensor).forEach(ReferenceCounting::freeRef);
    return CudaTensor.wrap(outputPtr, passbackDescriptor, precision);
  }
  
  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @param offset                the offset
   * @return the int [ ]
   */
  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i ->
      Math.min(sourceDimensions[i], destinationDimensions[i] + offset[i]) -
        Math.max(offset[i], 0)
    );
    return viewDim;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
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
