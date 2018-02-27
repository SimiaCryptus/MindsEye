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

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgCropLayer extends LayerBase implements MultiPrecision<ImgCropLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgCropLayer.class);
  
  private int sizeX;
  private int sizeY;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgCropLayer() {
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgCropLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    assert 0 < sizeX;
    assert 0 < sizeY;
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgCropLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    assert 0 < sizeX;
    assert 0 < sizeY;
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgCropLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    assert 1 == inObj.length;
    final Result in = inObj[0];
    assert 3 == in.getData().getDimensions().length;
    final int length = in.getData().length();
    @Nonnull int[] dimIn = in.getData().getDimensions();
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      in.addRef();
      in.getData().addRef();
      return in;
    }
    @javax.annotation.Nonnull final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final TensorList outputData = CudaSystem.eval(gpu -> {
      @Nullable final CudaTensor inputBuffer = gpu.getTensor(in.getData(), precision, MemoryType.Device);
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      @javax.annotation.Nonnull final CudaMemory outputBuffer = gpu.allocate((long) length * dimOut[2] * dimOut[1] * dimOut[0] * precision.size, MemoryType.Managed, dirty);
      CudaDevice.CudaTensorDescriptor descriptorCudaResource = copy(gpu, length, dimIn, inputBuffer, dimOut, outputBuffer);
      Arrays.stream(new ReferenceCounting[]{inputBuffer}).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(CudaTensor.wrap(outputBuffer, descriptorCudaResource), length, dimOut, precision);
    });
    return new Result(outputData, (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList error) -> {
      if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
        throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
      }
      if (error.length() != outputData.length()) {
        throw new AssertionError(error.length() + " != " + outputData.length());
      }
      assert error.length() == in.getData().length();
      if (in.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
          @Nullable final CudaTensor errorPtr = gpu.getTensor(error, precision, MemoryType.Device);
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          @javax.annotation.Nonnull final CudaMemory passbackBuffer = gpu.allocate((long) (length * dimIn[2] * dimIn[1] * dimIn[0] * precision.size), MemoryType.Managed, dirty);
          CudaDevice.CudaTensorDescriptor descriptorCudaResource = copy(gpu, length, dimOut, errorPtr, dimIn, passbackBuffer);
          Arrays.stream(new ReferenceCounting[]{errorPtr}).forEach(ReferenceCounting::freeRef);
          return CudaTensorList.wrap(CudaTensor.wrap(passbackBuffer, descriptorCudaResource), length, dimIn, precision);
        });
        in.accumulate(buffer, passbackTensorList);
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
   * @param sourceDimensions      the length in
   * @param source                the input buffer
   * @param destinationDimensions the length out
   * @param destination           the output buffer
   * @return the cuda resource
   */
  public CudaDevice.CudaTensorDescriptor copy(@Nonnull CudnnHandle gpu, int length, @Nonnull int[] sourceDimensions, @Nonnull CudaTensor source, @Nonnull int[] destinationDimensions, @Nonnull CudaMemory destination) {
    if (3 != sourceDimensions.length) throw new IllegalArgumentException("inputDimensions.length");
    if (3 != destinationDimensions.length) throw new IllegalArgumentException("dimOut.length");
    if (sourceDimensions[2] != destinationDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", sourceDimensions[2], destinationDimensions[2]));
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @javax.annotation.Nonnull final int[] viewDim = getViewDimensions(sourceDimensions, destinationDimensions);
    @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor sourceViewDescriptor = gpu.newTensorDescriptor(
      precision.code,//
      length,//
      viewDim[2],//
      viewDim[1],//
      viewDim[0],//
      source.descriptor.nStride,//
      source.descriptor.cStride,//
      source.descriptor.hStride,//
      source.descriptor.wStride);
    @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor destinationViewDescriptor = gpu.newTensorDescriptor(
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
  
    if (sourceDimensions[0] < destinationDimensions[0]) {
      destinationOffset += (destinationDimensions[0] - sourceDimensions[0]) / 2;
    }
    else {
      sourceOffset += (sourceDimensions[0] - destinationDimensions[0]) / 2;
    }
    if (sourceDimensions[1] < destinationDimensions[1]) {
      destinationOffset += destinationDimensions[0] * ((destinationDimensions[1] - sourceDimensions[1]) / 2);
    }
    else {
      sourceOffset += sourceDimensions[0] * ((sourceDimensions[1] - destinationDimensions[1]) / 2);
    }
  
    assert sourceOffset >= 0;
    assert destinationOffset >= 0;
    assert sourceOffset + Tensor.length(viewDim) <= Tensor.length(sourceDimensions);
    assert destinationOffset + Tensor.length(viewDim) <= Tensor.length(destinationDimensions);
  
    CudaSystem.handle(gpu.cudnnTransformTensor(
      precision.getPointer(1.0),
      sourceViewDescriptor.getPtr(), source.memory.getPtr().withByteOffset(sourceOffset * precision.size),
      precision.getPointer(0.0),
      destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(destinationOffset * precision.size)
    ));
    Arrays.stream(new ReferenceCounting[]{sourceViewDescriptor}).forEach(ReferenceCounting::freeRef);
    return destinationViewDescriptor;
    
  }
  
  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @return the int [ ]
   */
  @javax.annotation.Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions) {
    @javax.annotation.Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> Math.min(sourceDimensions[i], destinationDimensions[i]));
    return viewDim;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ImgCropLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
