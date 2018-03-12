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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudnnHandle;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
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
  protected ImgCropLayer(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
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
  public static ImgCropLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
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
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      input.addRef();
      input.getData().addRef();
      return input;
    }
    @Nonnull final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final TensorList outputData = CudaSystem.eval(gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(input.getData(), precision, MemoryType.Device, false);
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      CudaTensor cudaTensor = copy(gpu, inputTensor, length, dimIn, dimOut, dirty);
      Stream.<ReferenceCounting>of(inputTensor).forEach(ReferenceCounting::freeRef);
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
          @Nullable final CudaTensor errorPtr = gpu.getTensor(error, precision, MemoryType.Device, false);
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          CudaTensor cudaTensor = ImgCropLayer.this.copy(gpu, errorPtr, length, dimOut, dimIn, dirty);
          Stream.<ReferenceCounting>of(errorPtr).forEach(ReferenceCounting::freeRef);
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
  
  /**
   * Copy cuda tensor.
   *
   * @param gpu              the gpu
   * @param inputTensor      the input tensor
   * @param length           the length
   * @param inputDimensions  the input dimensions
   * @param outputDimensions the output dimensions
   * @param dirty            the dirty
   * @return the cuda tensor
   */
  public CudaTensor copy(final CudnnHandle gpu, final CudaTensor inputTensor, final int length, final int[] inputDimensions, final int[] outputDimensions, final boolean dirty) {
    if (3 != inputDimensions.length) throw new IllegalArgumentException("inputDimensions.length");
    if (3 != outputDimensions.length) throw new IllegalArgumentException("dimOut.length");
    if (inputDimensions[2] != outputDimensions[2]) {
      throw new IllegalArgumentException(String.format("%d != %d", inputDimensions[2], outputDimensions[2]));
    }
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(inputDimensions, outputDimensions);
    int sourceOffset = 0;
    int destinationOffset = 0;
    if (inputDimensions[0] < outputDimensions[0]) {
      destinationOffset += (outputDimensions[0] - inputDimensions[0]) / 2;
    }
    else {
      sourceOffset += (inputDimensions[0] - outputDimensions[0]) / 2;
    }
    if (inputDimensions[1] < outputDimensions[1]) {
      destinationOffset += outputDimensions[0] * ((outputDimensions[1] - inputDimensions[1]) / 2);
    }
    else {
      sourceOffset += inputTensor.descriptor.hStride * ((inputDimensions[1] - outputDimensions[1]) / 2);
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
    CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
    try {
      if (Arrays.equals(viewDim, outputDimensions)) {
        assert sourceOffset >= 0;
        assert destinationOffset == 0;
        return CudaTensor.wrap(inputTensorMemory.withByteOffset(sourceOffset * precision.size), sourceViewDescriptor, precision);
      }
      
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
      @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size, MemoryType.Managed.normalize(), dirty);
      CudaSystem.handle(gpu.cudnnTransformTensor(
        precision.getPointer(1.0),
        sourceViewDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size),
        precision.getPointer(0.0),
        destinationViewDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(destinationOffset * precision.size)
      ));
      assert gpu.getDeviceId() == CudaSystem.getThreadDeviceId();
      inputTensorMemory.dirty();
      outputBuffer.dirty();
      Stream.<ReferenceCounting>of(sourceViewDescriptor, destinationViewDescriptor).forEach(ReferenceCounting::freeRef);
      CudaDevice.CudaTensorDescriptor descriptorCudaResource = gpu.newTensorDescriptor(
        precision,//
        length,//
        outputDimensions[2],//
        outputDimensions[1],//
        outputDimensions[0],//
        outputDimensions[2] * outputDimensions[1] * outputDimensions[0],//
        outputDimensions[1] * outputDimensions[0],//
        outputDimensions[0],//
        1);
      return CudaTensor.wrap(outputBuffer, descriptorCudaResource, precision);
    } finally {
      inputTensorMemory.freeRef();
    }
  }
  
  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @return the int [ ]
   */
  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> Math.min(sourceDimensions[i], destinationDimensions[i]));
    return viewDim;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
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
  public ImgCropLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
