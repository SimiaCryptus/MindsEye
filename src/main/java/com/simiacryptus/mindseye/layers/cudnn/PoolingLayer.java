/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import jcuda.jcudnn.*;

import java.util.Arrays;
import java.util.List;

/**
 * The type Pooling layer.
 */
@SuppressWarnings("serial")
public class PoolingLayer extends NNLayer implements LayerPrecision<PoolingLayer> {
  
  private PoolingMode mode = PoolingMode.Max;
  private int paddingX = 0;
  private int paddingY = 0;
  private Precision precision = Precision.Double;
  private int strideX = 2;
  private int strideY = 2;
  private int windowX = 2;
  private int windowY = 2;
  
  /**
   * Instantiates a new Pooling layer.
   */
  public PoolingLayer() {
    super();
  }
  
  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected PoolingLayer(final JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    windowX = json.get("windowX").getAsInt();
    windowY = json.get("windowY").getAsInt();
    paddingX = json.get("paddingX").getAsInt();
    paddingY = json.get("paddingY").getAsInt();
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json pooling layer.
   *
   * @param json the json
   * @return the pooling layer
   */
  public static PoolingLayer fromJson(final JsonObject json) {
    return new PoolingLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final int poolDims = 2;
    final int windowSize[] = {windowX, windowY};
    final int padding[] = {paddingX, paddingY};
    final int stride[] = {strideX, strideY};
    try {
      ((CudaExecutionContext) nncontext).initThread();
      //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
      final NNResult input = inObj[0];
      final TensorList batch = input.getData();
      final int[] inputSize = batch.get(0).getDimensions();
      final int length = batch.length();
      final int inputDims = Tensor.dim(inputSize);
      final CudaResource<cudnnPoolingDescriptor> poolingDesc = CuDNN.createPoolingDescriptor(
        mode.id, poolDims, windowSize, padding, stride);
      final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      final int[] outputSize = new int[4];
      CuDNN.handle(CuDNN.cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(), inputDescriptor.getPtr(), 4, outputSize));
      assert inputSize[2] == outputSize[1];
      final CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[0], outputSize[1], outputSize[2], outputSize[3]);
      final CudaPtr alpha = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      final CudaPtr beta = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0);
      final CudaPtr inputData = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, batch);
      final CudaPtr outputData = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), precision.size * 1l * Tensor.dim(outputSize));
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      CuDNN.handle(CuDNN.cudnnPoolingForward(cudnnHandle, poolingDesc.getPtr(),
        alpha.getPtr(),
        inputDescriptor.getPtr(), inputData.getPtr(),
        beta.getPtr(),
        outputDescriptor.getPtr(), outputData.getPtr()));
      final TensorList output = new GpuTensorList(outputData, length, new int[]{outputSize[3], outputSize[2], outputSize[1]}, cudnnHandle, precision);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert error.length() == batch.length();
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          if (input.isAlive()) {
            final CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), inputDims * 1l * precision.size * length);
            CuDNN.handle(CuDNN.cudnnPoolingBackward(cudnnHandle, poolingDesc.getPtr(),
              alpha.getPtr(),
              outputDescriptor.getPtr(), outputData.getPtr(),
              outputDescriptor.getPtr(), errorPtr.getPtr(),
              inputDescriptor.getPtr(), inputData.getPtr(),
              beta.getPtr(),
              inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            input.accumulate(buffer, new GpuTensorList(passbackBuffer, length, inputSize, cudnnHandle, precision));
          }
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (final Throwable e) {
      throw new ComponentException("Error", e);
    }
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("windowX", windowX);
    json.addProperty("windowY", windowY);
    json.addProperty("paddingX", paddingX);
    json.addProperty("paddingY", paddingY);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
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
  public PoolingLayer setPaddingY(final int paddingY) {
    this.paddingY = paddingY;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
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
  public PoolingLayer setWindowY(final int windowY) {
    this.windowY = windowY;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
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
