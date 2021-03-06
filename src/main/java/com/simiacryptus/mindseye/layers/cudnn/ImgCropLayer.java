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
import java.util.UUID;
import java.util.stream.Stream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output png will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgCropLayer extends LayerBase implements MultiPrecision<ImgCropLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgCropLayer.class);

  private int sizeX;
  private int sizeY;
  private Precision precision = Precision.Double;

  /**
   * Instantiates a new Img eval key.
   */
  private ImgCropLayer() {
  }

  /**
   * Instantiates a new Img crop key.
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
   * Instantiates a new Img eval key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgCropLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
    assert 0 < sizeX;
    assert 0 < sizeY;
  }

  /**
   * From json img eval key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval key
   */
  public static ImgCropLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }

  /**
   * Copy cuda tensor.
   *
   * @param gpu              the gpu
   * @param input            the input tensor
   * @param length           the length
   * @param inputDimensions  the input dimensions
   * @param outputDimensions the output dimensions
   * @param dirty            the dirty
   * @param precision        the precision
   * @return the cuda tensor
   */
  public static CudaTensor copy(final CudnnHandle gpu, final CudaTensor input, final int length, final int[] inputDimensions, final int[] outputDimensions, final boolean dirty, Precision precision) {
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
    } else {
      sourceOffset += (inputDimensions[0] - outputDimensions[0]) / 2;
    }
    if (inputDimensions[1] < outputDimensions[1]) {
      destinationOffset += outputDimensions[0] * ((outputDimensions[1] - inputDimensions[1]) / 2);
    } else {
      sourceOffset += input.descriptor.hStride * ((inputDimensions[1] - outputDimensions[1]) / 2);
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
        input.descriptor.nStride,//
        input.descriptor.cStride,//
        input.descriptor.hStride,//
        input.descriptor.wStride);
    CudaMemory inputTensorMemory = input.getMemory(gpu);
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
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
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
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> Math.min(sourceDimensions[i], destinationDimensions[i]));
    return viewDim;
  }

  /**
   * Gets compatibility key.
   *
   * @return the compatibility key
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull int[] dimIn = inputData.getDimensions();
    if (dimIn[0] == sizeX && dimIn[1] == sizeY) {
      return input;
    }
    @Nonnull final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final TensorList outputData = CudaSystem.run(gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      inputData.freeRef();
      boolean dirty = dimOut[0] <= dimIn[0] && dimOut[1] <= dimIn[1];
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      CudaTensor cudaTensor = copy(gpu, inputTensor, length, dimIn, dimOut, dirty, precision);
      Stream.<ReferenceCounting>of(inputTensor).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(cudaTensor, length, dimOut, precision);
    }, inputData);
    int[] output_dimensions = outputData.getDimensions();
    int output_length = outputData.length();
    return new Result(outputData, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), output_dimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(output_dimensions));
      }
      if (delta.length() != output_length) {
        throw new AssertionError(delta.length() + " != " + output_length);
      }
      assert delta.length() == length;


      if (input.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          @Nullable final CudaTensor errorPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
          delta.freeRef();
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          CudaTensor cudaTensor = copy(gpu, errorPtr, length, dimOut, dimIn, dirty, precision);
          Stream.<ReferenceCounting>of(errorPtr).forEach(ReferenceCounting::freeRef);
          return CudaTensorList.wrap(cudaTensor, length, dimIn, precision);
        }, delta);
        input.accumulate(buffer, passbackTensorList);
      } else {
        delta.freeRef();
      }


    }) {

      @Override
      public void accumulate(final DeltaSet<UUID> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }

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

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
