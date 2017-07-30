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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnActivationMode.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class DirectActivationLayer extends DirectCuDNNLayer {


  public enum Mode {
    RELU(CUDNN_ACTIVATION_RELU),
    SIGMOID(CUDNN_ACTIVATION_SIGMOID);
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

  protected DirectActivationLayer(JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
  }

  final int mode;

  public DirectActivationLayer(Mode mode) {
    this(mode.id);
  }

  public DirectActivationLayer(int id) {
    this.mode = id;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    int[] outputSize = inputSize;
    int length = batch.length();
    int inputDims = Tensor.dim(inputSize);

    try {

      CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CuDNN.CuDNNPtr alpha = CuDNN.javaPtr(1.0);
      CuDNN.CuDNNPtr beta = CuDNN.javaPtr(0.0);

      CuDNN.CuDNNPtr inputData = toDevice(batch);
      CuDNN.CuDNNPtr outputData = CuDNN.alloc(Sizeof.DOUBLE * inputDims);
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
      TensorList output = fromDevice(outputData, length, outputSize);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          assert (error.length() == batch.length());
          assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          CuDNN.CuDNNPtr errorPtr = toDevice(error);
          if (input.isAlive()) {
            CuDNN.CuDNNPtr passbackBuffer = CuDNN.alloc(inputDims * Sizeof.DOUBLE);
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
            input.accumulate(buffer, fromDevice(passbackBuffer, length, inputSize));
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
