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
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;

import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnActivationBackward;
import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnActivationForward;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Activation layer.
 */
public class ActivationLayer extends NNLayer implements LayerPrecision<ActivationLayer> {
  /**
   * The Mode.
   */
  final int mode;
  private Precision precision = Precision.Double;
  
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected ActivationLayer(JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param mode the mode
   */
  public ActivationLayer(Mode mode) {
    this(mode.id);
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param id the id
   */
  public ActivationLayer(int id) {
    this.mode = id;
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @return the activation layer
   */
  public static ActivationLayer fromJson(JsonObject json) {
    return new ActivationLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    int[] outputSize = inputSize;
    int length = batch.length();
    int inputDims = Tensor.dim(inputSize);
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaPtr alpha = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      CudaPtr beta = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0);
      
      CudaPtr inputData = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, batch);
      CudaPtr outputData = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), precision.size * 1l * inputDims * length);
      CudaResource<cudnnActivationDescriptor> activationDesc = CuDNN.newActivationDescriptor(mode, CUDNN_NOT_PROPAGATE_NAN, 0);
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      try {
        CuDNN.handle(cudnnActivationForward(((CuDNN) nncontext).cudnnHandle, activationDesc.getPtr(),
          alpha.getPtr(),
          inputDescriptor.getPtr(), inputData.getPtr(),
          beta.getPtr(),
          inputDescriptor.getPtr(), outputData.getPtr()));
      } catch (Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
      }
      TensorList output = new GpuTensorList(outputData, length, outputSize, ((CuDNN) nncontext).cudnnHandle, precision);
      //assert output.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          //assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          ((CudaExecutionContext) nncontext).initThread();
          CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          if (input.isAlive()) {
            CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), inputDims * 1l * precision.size * length);
            try {
              CuDNN.handle(cudnnActivationBackward(((CuDNN) nncontext).cudnnHandle, activationDesc.getPtr(),
                alpha.getPtr(),
                inputDescriptor.getPtr(), outputData.getPtr(),
                inputDescriptor.getPtr(), errorPtr.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                beta.getPtr(),
                inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            input.accumulate(buffer, new GpuTensorList(passbackBuffer, length, inputSize, cudnnHandle, precision));
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ActivationLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }
  
  
  /**
   * The enum Mode.
   */
  public enum Mode {
    /**
     * Relu mode.
     */
    RELU(CUDNN_ACTIVATION_RELU),
    /**
     * Sigmoid mode.
     */
    SIGMOID(CUDNN_ACTIVATION_SIGMOID);
    /**
     * The Id.
     */
    public final int id;
    
    Mode(int id) {
      this.id = id;
    }
  }
  
}
