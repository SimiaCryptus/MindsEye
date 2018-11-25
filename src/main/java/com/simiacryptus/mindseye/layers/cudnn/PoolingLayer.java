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
import com.simiacryptus.mindseye.layers.java.AvgPoolingLayer;
import com.simiacryptus.mindseye.layers.java.MaxPoolingLayer;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnPoolingMode;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * The standard png-pixel pooling key. Using a configurable stride and window size, reduces pixels using either the
 * Max or Avg operation.
 */
@SuppressWarnings("serial")
public class PoolingLayer extends LayerBase implements MultiPrecision<PoolingLayer> {

  private PoolingMode mode = PoolingMode.Max;
  private int paddingX = 0;
  private int paddingY = 0;
  private Precision precision = Precision.Double;
  private int strideX = 2;
  private int strideY = 2;
  private int windowX = 2;
  private int windowY = 2;
  private double alpha;

  /**
   * Instantiates a new Pooling key.
   */
  public PoolingLayer() {
    super();
    alpha = 1.0;
  }

  /**
   * Instantiates a new Pooling key.
   *
   * @param json the json
   */
  protected PoolingLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    alpha = json.get("alpha").getAsDouble();
    windowX = json.get("windowX").getAsInt();
    windowY = json.get("windowY").getAsInt();
    paddingX = json.get("paddingX").getAsInt();
    paddingY = json.get("paddingY").getAsInt();
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = 1.0;
  }

  /**
   * From json pooling key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the pooling key
   */
  public static PoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PoolingLayer(json);
  }

  /**
   * Gets compatibility key.
   *
   * @return the compatibility key
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == PoolingMode.Max) return this.as(MaxPoolingLayer.class);
    if (mode == PoolingMode.Avg) return this.as(AvgPoolingLayer.class);
    else throw new RuntimeException("Not Implemented");
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    final int poolDims = 2;
    @Nonnull final int windowSize[] = {windowX, windowY};
    @Nonnull final int padding[] = {paddingX, paddingY};
    @Nonnull final int stride[] = {strideX, strideY};
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    @Nonnull final int[] outputSize = new int[4];
    final CudaTensor outputData = CudaSystem.run(gpu -> {
      try {
        gpu.initThread();
        @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(
            mode.id, poolDims, windowSize, padding, stride);
        @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        CudaSystem.handle(CudaSystem.cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(), inputTensor.descriptor.getPtr(), 4, outputSize));
        assert inputSize[2] == outputSize[1];
        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, outputSize[0], outputSize[1], outputSize[2], outputSize[3], outputSize[1] * outputSize[2] * outputSize[3], outputSize[2] * outputSize[3], outputSize[3], 1);
        @Nonnull final CudaMemory outputTensor = gpu.allocate((long) precision.size * Tensor.length(outputSize), MemoryType.Managed.normalize(), true);
        CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
        CudaSystem.handle(gpu.cudnnPoolingForward(poolingDesc.getPtr(),
            precision.getPointer(alpha),
            inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
            precision.getPointer(0.0),
            outputDescriptor.getPtr(), outputTensor.getPtr()));
        assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
        inputDataMemory.dirty();
        outputTensor.dirty();
        Stream.<ReferenceCounting>of(inputTensor, poolingDesc, inputDataMemory).forEach(ReferenceCounting::freeRef);
        return CudaTensor.wrap(outputTensor, outputDescriptor, precision);
      } catch (@Nonnull final Throwable e) {
        throw new ComponentException("Error", e);
      }
    }, inputData);
    return new Result(CudaTensorList.create(outputData, length, new int[]{outputSize[3], outputSize[2], outputSize[1]}, precision),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
          assert error.length() == inputData.length();
          if (input.isAlive()) {
            TensorList data = CudaSystem.run(gpu -> {
              @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision,
                  length, inputSize[2], inputSize[1], inputSize[0],
                  inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1);
              @Nonnull final CudaResource<cudnnPoolingDescriptor> poolingDesc = gpu.createPoolingDescriptor(
                  mode.id, poolDims, windowSize, padding, stride);
              @Nullable final CudaTensor inputTensor;
              synchronized (gpu) {
                inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
              }
              @Nullable final CudaTensor errorPtr;
              synchronized (gpu) {
                errorPtr = gpu.getTensor(error, precision, MemoryType.Device, true);
              }
              @Nonnull final CudaMemory passbackBuffer = gpu.allocate((long) inputDims * precision.size * length, MemoryType.Managed.normalize(), true);
              CudaMemory outputDataMemory = outputData.getMemory(gpu);
              CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
              CudaMemory inputDataMemory = inputTensor.getMemory(gpu);
              CudaSystem.handle(gpu.cudnnPoolingBackward(poolingDesc.getPtr(),
                  precision.getPointer(this.alpha), outputData.descriptor.getPtr(), outputDataMemory.getPtr(),
                  errorPtr.descriptor.getPtr(), errorPtrMemory.getPtr(),
                  inputTensor.descriptor.getPtr(), inputDataMemory.getPtr(),
                  precision.getPointer(0.0), passbackDescriptor.getPtr(), passbackBuffer.getPtr()));
              outputDataMemory.dirty();
              errorPtrMemory.dirty();
              inputDataMemory.dirty();
              passbackBuffer.dirty();

              Stream.<ReferenceCounting>of(errorPtr, inputTensor, poolingDesc, outputDataMemory, errorPtrMemory, inputDataMemory).forEach(ReferenceCounting::freeRef);
              return CudaTensorList.wrap(CudaTensor.wrap(passbackBuffer, passbackDescriptor, precision), length, inputSize, precision);
            }, error);
            input.accumulate(buffer, data);
          }
        }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        inputData.freeRef();
        outputData.freeRef();
      }

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("windowX", windowX);
    json.addProperty("windowY", windowY);
    json.addProperty("paddingX", paddingX);
    json.addProperty("paddingY", paddingY);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("alpha", alpha);
    json.addProperty("precision", precision.name());
    return json;
  }

  /**
   * Gets mode.
   *
   * @return the mode
   */
  public PoolingMode getMode() {
    return mode;
  }

  /**
   * Sets mode.
   *
   * @param mode the mode
   * @return the mode
   */
  @Nonnull
  public PoolingLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this;
  }

  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }

  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  @Nonnull
  public PoolingLayer setPaddingX(final int paddingX) {
    this.paddingX = paddingX;
    return this;
  }

  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }

  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  @Nonnull
  public PoolingLayer setPaddingY(final int paddingY) {
    this.paddingY = paddingY;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public PoolingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  /**
   * Gets stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }

  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   * @return the stride x
   */
  @Nonnull
  public PoolingLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this;
  }

  /**
   * Gets stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }

  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   * @return the stride y
   */
  @Nonnull
  public PoolingLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this;
  }

  /**
   * Gets window x.
   *
   * @return the window x
   */
  public int getWindowX() {
    return windowX;
  }

  /**
   * Sets window x.
   *
   * @param windowX the window x
   * @return the window x
   */
  @Nonnull
  public PoolingLayer setWindowX(final int windowX) {
    this.windowX = windowX;
    return this;
  }

  /**
   * Gets window y.
   *
   * @return the window y
   */
  public int getWindowY() {
    return windowY;
  }

  /**
   * Sets window y.
   *
   * @param windowY the window y
   * @return the window y
   */
  @Nonnull
  public PoolingLayer setWindowY(final int windowY) {
    this.windowY = windowY;
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  /**
   * Sets window xy.
   *
   * @param x the x
   * @param y the y
   * @return the window xy
   */
  @Nonnull
  public PoolingLayer setWindowXY(int x, int y) {
    setWindowY(y);
    setWindowX(x);
    return this;
  }

  /**
   * Sets stride xy.
   *
   * @param x the x
   * @param y the y
   * @return the stride xy
   */
  @Nonnull
  public PoolingLayer setStrideXY(int x, int y) {
    setStrideX(x);
    setStrideY(y);
    return this;
  }

  /**
   * Sets padding xy.
   *
   * @param x the x
   * @param y the y
   * @return the padding xy
   */
  @Nonnull
  public PoolingLayer setPaddingXY(int x, int y) {
    setPaddingX(x);
    setPaddingY(y);
    return this;
  }

  /**
   * Gets alphaList.
   *
   * @return the alphaList
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alphaList.
   *
   * @param alpha the alphaList
   * @return the alphaList
   */
  public PoolingLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  /**
   * The enum Pooling mode.
   */
  public enum PoolingMode {
    /**
     * Avg pooling mode.
     */
    Avg(cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
    /**
     * Max pooling mode.
     */
    Max(cudnnPoolingMode.CUDNN_POOLING_MAX);
    /**
     * The Id.
     */
    final int id;

    PoolingMode(final int id) {
      this.id = id;
    }
  }
}
