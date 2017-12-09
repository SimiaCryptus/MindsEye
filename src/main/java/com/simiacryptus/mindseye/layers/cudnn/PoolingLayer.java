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
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;

import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnGetPoolingNdForwardOutputDim;
import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnPoolingBackward;
import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnPoolingForward;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Pooling layer.
 */
public class PoolingLayer extends NNLayer {
  
  private PoolingMode mode = PoolingMode.Max;
  private int windowX = 2;
  private int windowY = 2;
  private int paddingX = 0;
  private int paddingY = 0;
  private int strideX = 2;
  private int strideY = 2;
  
  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected PoolingLayer(JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    windowX = json.get("windowX").getAsInt();
    windowY = json.get("windowY").getAsInt();
    paddingX = json.get("paddingX").getAsInt();
    paddingY = json.get("paddingY").getAsInt();
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
  }
  
  /**
   * Instantiates a new Pooling layer.
   */
  public PoolingLayer() {
    super();
  }
  
  /**
   * From json pooling layer.
   *
   * @param json the json
   * @return the pooling layer
   */
  public static PoolingLayer fromJson(JsonObject json) {
    return new PoolingLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("windowX", windowX);
    json.addProperty("windowY", windowY);
    json.addProperty("paddingX", paddingX);
    json.addProperty("paddingY", paddingY);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
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
      int length = batch.length();
      int inputDims = Tensor.dim(inputSize);
      CudaResource<cudnnPoolingDescriptor> poolingDesc = CuDNN.createPoolingDescriptor(
        mode.id, poolDims, windowSize, padding, stride);
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      int[] outputSize = new int[4];
      CuDNN.handle(cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(), inputDescriptor.getPtr(), 4, outputSize));
      assert (inputSize[2] == outputSize[1]);
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[0], outputSize[1], outputSize[2], outputSize[3]);
      CudaPtr alpha = CudaPtr.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), Precision.Double, 1.0);
      CudaPtr beta = CudaPtr.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), Precision.Double, 0.0);
      CudaPtr inputData = CudaPtr.toDevice(((CudaExecutionContext) nncontext).getDeviceNumber(), batch, Precision.Double);
      CudaPtr outputData = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Sizeof.DOUBLE * 1l * Tensor.dim(outputSize));
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      CuDNN.handle(cudnnPoolingForward(cudnnHandle, poolingDesc.getPtr(),
        alpha.getPtr(),
        inputDescriptor.getPtr(), inputData.getPtr(),
        beta.getPtr(),
        outputDescriptor.getPtr(), outputData.getPtr()));
      TensorList output = CudaPtr.fromDevice(outputData, length, new int[]{outputSize[3], outputSize[2], outputSize[1]}, cudnnHandle, Precision.Double);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          CudaPtr errorPtr = CudaPtr.toDevice(((CudaExecutionContext) nncontext).getDeviceNumber(), error, Precision.Double);
          if (input.isAlive()) {
            CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), inputDims * 1l * Sizeof.DOUBLE * length);
            CuDNN.handle(cudnnPoolingBackward(cudnnHandle, poolingDesc.getPtr(),
              alpha.getPtr(),
              outputDescriptor.getPtr(), outputData.getPtr(),
              outputDescriptor.getPtr(), errorPtr.getPtr(),
              inputDescriptor.getPtr(), inputData.getPtr(),
              beta.getPtr(),
              inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            input.accumulate(buffer, CudaPtr.fromDevice(passbackBuffer, length, inputSize, cudnnHandle, Precision.Double));
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error", e);
    }
  }
  
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
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
  public PoolingLayer setMode(PoolingMode mode) {
    this.mode = mode;
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
   */
  public void setWindowX(int windowX) {
    this.windowX = windowX;
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
   */
  public void setWindowY(int windowY) {
    this.windowY = windowY;
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
   */
  public void setPaddingX(int paddingX) {
    this.paddingX = paddingX;
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
   */
  public void setPaddingY(int paddingY) {
    this.paddingY = paddingY;
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
   */
  public void setStrideX(int strideX) {
    this.strideX = strideX;
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
   */
  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }
  
  /**
   * The enum Pooling mode.
   */
  public enum PoolingMode {
    /**
     * Max pooling mode.
     */
    Max(CUDNN_POOLING_MAX),
    /**
     * Avg pooling mode.
     */
    Avg(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
    /**
     * The Id.
     */
    final int id;
    
    PoolingMode(int id) {
      this.id = id;
    }
  }
}
