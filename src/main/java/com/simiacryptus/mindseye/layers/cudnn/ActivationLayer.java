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
 * The type Activation layer.
 */
@SuppressWarnings("serial")
public class ActivationLayer extends NNLayer implements LayerPrecision<ActivationLayer> {
  /**
   * The Mode.
   */
  final int mode;
  private Precision precision = Precision.Double;
  
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param id the id
   */
  public ActivationLayer(final int id) {
    mode = id;
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected ActivationLayer(final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param mode the mode
   */
  public ActivationLayer(final Mode mode) {
    this(mode.id);
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @return the activation layer
   */
  public static ActivationLayer fromJson(final JsonObject json) {
    return new ActivationLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    final int[] outputSize = inputSize;
    final int length = batch.length();
    final int inputDims = Tensor.dim(inputSize);
    
    try {
  
      final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      final CudaPtr alpha = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      final CudaPtr beta = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0);
  
      final CudaPtr inputData = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, batch);
      final CudaPtr outputData = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), precision.size * 1l * inputDims * length);
      final CudaResource<cudnnActivationDescriptor> activationDesc = CuDNN.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      try {
        CuDNN.handle(CuDNN.cudnnActivationForward(((CuDNN) nncontext).cudnnHandle, activationDesc.getPtr(),
          alpha.getPtr(),
          inputDescriptor.getPtr(), inputData.getPtr(),
          beta.getPtr(),
          inputDescriptor.getPtr(), outputData.getPtr()));
      } catch (final Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
      }
      final TensorList output = new GpuTensorList(outputData, length, outputSize, ((CuDNN) nncontext).cudnnHandle, precision);
      //assert output.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          //assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          ((CudaExecutionContext) nncontext).initThread();
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          if (input.isAlive()) {
            final CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), inputDims * 1l * precision.size * length);
            try {
              CuDNN.handle(CuDNN.cudnnActivationBackward(((CuDNN) nncontext).cudnnHandle, activationDesc.getPtr(),
                alpha.getPtr(),
                inputDescriptor.getPtr(), outputData.getPtr(),
                inputDescriptor.getPtr(), errorPtr.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                beta.getPtr(),
                inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            } catch (final Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            final GpuTensorList data = new GpuTensorList(passbackBuffer, length, inputSize, cudnnHandle, precision);
            input.accumulate(buffer, data);
            data.recycle();
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (final Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  
  /**
   * The enum Mode.
   */
  public enum Mode {
    /**
     * Relu mode.
     */
    RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU),
    /**
     * Sigmoid mode.
     */
    SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID);
    /**
     * The Id.
     */
    public final int id;
  
    Mode(final int id) {
      this.id = id;
    }
  }
  
}
