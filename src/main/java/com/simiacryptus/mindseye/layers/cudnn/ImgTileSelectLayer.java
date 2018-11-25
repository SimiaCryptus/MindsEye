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
public class ImgTileSelectLayer extends LayerBase implements MultiPrecision<ImgTileSelectLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileSelectLayer.class);
  private int positionX;
  private int positionY;

  private int sizeY;
  private int sizeX;
  private Precision precision = Precision.Double;

  /**
   * Instantiates a new Img eval key.
   */
  private ImgTileSelectLayer() {
  }

  /**
   * Instantiates a new Img crop key.
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
   * Instantiates a new Img eval key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileSelectLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeY = json.get("sizeX").getAsInt();
    sizeX = json.get("sizeY").getAsInt();
    positionX = json.get("positionX").getAsInt();
    positionY = json.get("positionY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }

  /**
   * From json img eval key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval key
   */
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json, rs);
  }

  /**
   * Copy cuda tensor.
   *
   * @param gpu              the gpu
   * @param input            the input
   * @param inputDimensions  the input dimensions
   * @param outputDimensions the output dimensions
   * @param precision        the precision
   * @param positionX        the position x
   * @param positionY        the position y
   * @param dirty            the dirty
   * @return the cuda tensor
   */
  public static CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions, final int[] outputDimensions, Precision precision, final int positionX, final int positionY, final boolean dirty) {
    @Nonnull final CudaMemory outputPtr = gpu.allocate((long) input.length() * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size, MemoryType.Managed.normalize(), dirty);
    return copy(gpu, input, inputDimensions, outputDimensions, positionX, positionY, precision, outputPtr);
  }

  /**
   * Copy cuda tensor.
   *
   * @param gpu             the gpu
   * @param input           the input
   * @param inputDimensions the input dimensions
   * @param positionX       the position x
   * @param positionY       the position y
   * @param precision       the precision
   * @param output          the output
   * @return the cuda tensor
   */
  public static CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions, final int positionX, final int positionY, Precision precision, final CudaTensor output) {
    return copy(gpu, input, inputDimensions,
        new int[]{output.descriptor.width, output.descriptor.height, output.descriptor.channels},
        positionX, positionY, precision, output.getMemory(gpu));
  }

  /**
   * Copy cuda tensor.
   *
   * @param gpu              the gpu
   * @param input            the input
   * @param inputDimensions  the input dimensions
   * @param outputDimensions the output dimensions
   * @param positionX        the position x
   * @param positionY        the position y
   * @param precision        the precision
   * @param outputPtr        the output ptr
   * @return the cuda tensor
   */
  public static CudaTensor copy(final CudnnHandle gpu, @Nonnull final TensorList input, final int[] inputDimensions, final int[] outputDimensions, final int positionX, final int positionY, final Precision precision, final CudaMemory outputPtr) {
    final int length = input.length();
    if (3 != inputDimensions.length) throw new IllegalArgumentException("inputDimensions.length");
    if (3 != outputDimensions.length) throw new IllegalArgumentException("dimOut.length");
    int bands = inputDimensions[2];
    if (bands != outputDimensions[2])
      throw new IllegalArgumentException(String.format("%d != %d", bands, outputDimensions[2]));
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    @Nonnull final int[] viewDim = getViewDimensions(inputDimensions, outputDimensions, new int[]{positionX, positionY, 0});
    @Nullable final CudaTensor inputTensor = gpu.getTensor(input, precision, MemoryType.Device, false);
    int sourceOffset = 0;
    int destinationOffset = 0;
    if (positionX < 0) {
      destinationOffset += Math.abs(positionX);
    } else {
      sourceOffset += Math.abs(positionX);
    }
    if (positionY < 0) {
      destinationOffset += outputDimensions[0] * Math.abs((positionY));
    } else {
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
      CudaSystem.handle(gpu.cudnnTransformTensor(
          precision.getPointer(1.0),
          sourceViewDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset(sourceOffset * precision.size),
          precision.getPointer(1.0),
          destinationViewDescriptor.getPtr(), outputPtr.getPtr().withByteOffset(destinationOffset * precision.size)
      ));
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      outputPtr.dirty();
      inputTensorMemory.dirty();
      Stream.<ReferenceCounting>of(sourceViewDescriptor, destinationViewDescriptor).forEach(ReferenceCounting::freeRef);

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
    } finally {
      inputTensorMemory.freeRef();
    }
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
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i ->
        Math.min(sourceDimensions[i], destinationDimensions[i] + offset[i]) - Math.max(offset[i], 0)
    );
    return viewDim;
  }

  /**
   * Gets compatibility key.
   *
   * @return the compatibility key
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer.class);
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
    if (dimIn[0] == sizeY && dimIn[1] == sizeX) {
      return input;
    }
    @Nonnull final int[] dimOut = getViewDimensions(dimIn, new int[]{sizeY, sizeX, dimIn[2]}, new int[]{positionX, positionY, 0});
    final TensorList outputData = CudaSystem.run(gpu -> {
      assert dimOut[0] > 0;
      assert dimOut[1] > 0;
      assert dimOut[2] > 0;
      boolean dirty = dimOut[0] == dimIn[0] && dimOut[1] == dimIn[1];
      CudaTensor cudaTensor = copy(gpu, inputData, dimIn, dimOut, precision, this.positionX, this.positionY, dirty);
      return CudaTensorList.wrap(cudaTensor, length, dimOut, precision);
    }, inputData);
    int[] outputDimensions = outputData.getDimensions();
    assert length == outputData.length();
    return new Result(outputData, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
      if (!Arrays.equals(error.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      if (error.length() != length) {
        throw new AssertionError(error.length() + " != " + length);
      }
      assert error.length() == inputData.length();
      if (input.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          boolean dirty = dimOut[0] >= dimIn[0] && dimOut[1] >= dimIn[1];
          CudaTensor cudaTensor = copy(gpu, error, dimOut, dimIn, precision, -this.positionX, -this.positionY, dirty);
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

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
