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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.DirectCuDNNLayer;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

import java.util.Arrays;
import java.util.List;

import static jcuda.jcudnn.JCudnn.cudnnActivationBackward;
import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import jcuda.runtime.JCuda;

/**
 * The type Activation layer.
 */
public class ActivationLayer extends DirectCuDNNLayer {
  /**
   * From json activation layer.
   *
   * @param json the json
   * @return the activation layer
   */
  public static ActivationLayer fromJson(JsonObject json) {
    return new ActivationLayer(json);
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

    private Mode(int id) {
      this.id = id;
    }
  }

  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("mode",mode);
    return json;
  }
  
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
   * The Mode.
   */
  final int mode;
  
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

  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    JCuda.cudaSetDevice(nncontext.getCudaDeviceId());
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    int[] outputSize = inputSize;
    int length = batch.length();
    int inputDims = Tensor.dim(inputSize);

    try {

      CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CuDNN.CuDNNPtr alpha = CuDNN.javaPtr(1.0f);
      CuDNN.CuDNNPtr beta = CuDNN.javaPtr(0.0f);

      CuDNN.CuDNNPtr inputData = toDeviceAsFloat(batch);
      CuDNN.CuDNNPtr outputData = CuDNN.alloc(Sizeof.FLOAT * inputDims * length);
      CuDNN.CuDNNResource<cudnnActivationDescriptor> activationDesc = CuDNN.newActivationDescriptor(mode, CUDNN_PROPAGATE_NAN, 0);
      CuDNN.devicePool.with(device -> {
        try {
          CuDNN.handle(cudnnActivationForward(device.cudnnHandle, activationDesc.getPtr(),
                  alpha.getPtr(),
                  inputDescriptor.getPtr(), inputData.getPtr(),
                  beta.getPtr(),
                  inputDescriptor.getPtr(), outputData.getPtr()));
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(inputSize),e);
        }
      });
      TensorList output = fromDeviceFloat(outputData, length, outputSize);
      //assert output.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          //assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          JCuda.cudaSetDevice(nncontext.getCudaDeviceId());
          CuDNN.CuDNNPtr errorPtr = toDeviceAsFloat(error);
          if (input.isAlive()) {
            CuDNN.CuDNNPtr passbackBuffer = CuDNN.alloc(inputDims * Sizeof.FLOAT * length);
            try {
              CuDNN.devicePool.with(device -> {
                CuDNN.handle(cudnnActivationBackward(device.cudnnHandle, activationDesc.getPtr(),
                        alpha.getPtr(),
                        inputDescriptor.getPtr(), outputData.getPtr(),
                        inputDescriptor.getPtr(), errorPtr.getPtr(),
                        inputDescriptor.getPtr(), inputData.getPtr(),
                        beta.getPtr(),
                        inputDescriptor.getPtr(), passbackBuffer.getPtr()));
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error with " + Arrays.toString(inputSize),e);
            }
            input.accumulate(buffer, fromDeviceFloat(passbackBuffer, length, inputSize));
          }
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new RuntimeException("Error with image res " + Arrays.toString(inputSize),e);
    }
  }


  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
