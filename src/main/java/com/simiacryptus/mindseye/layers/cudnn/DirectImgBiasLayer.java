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
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class DirectImgBiasLayer extends DirectCuDNNLayer {

  public static DirectImgBiasLayer fromJson(JsonObject json) {
    return new DirectImgBiasLayer(json);
  }

  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }

  protected DirectImgBiasLayer(JsonObject json) {
    super(json);
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  private final double[] bias;

  public DirectImgBiasLayer(int bands) {
    this.bias = new double[bands];
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    assert(inputSize[2] == bias.length);
    int[] outputSize = inputSize;
    int length = batch.length();

    try {

      CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CuDNN.CuDNNResource<cudnnTensorDescriptor> filterDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
      CuDNN.CuDNNPtr alpha = CuDNN.javaPtr(1.0);
      CuDNN.CuDNNPtr beta = CuDNN.javaPtr(1.0);

      assert(0 < this.bias.length);
      CuDNN.CuDNNPtr filterPtr = CuDNN.write(this.bias);
      CuDNN.CuDNNPtr inputData = toDevice(batch);
      CuDNN.devicePool.with(device -> {
        try {
          CuDNN.handle(cudnnAddTensor(device.cudnnHandle, alpha.getPtr(),
                  filterDescriptor.getPtr(), filterPtr.getPtr(),
                  beta.getPtr(),
                  inputDescriptor.getPtr(), inputData.getPtr()));
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(inputSize),e);
        }
      });
      TensorList output = fromDevice(inputData, length, outputSize);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          CuDNN.CuDNNPtr errorPtr = toDevice(error);
          if (!isFrozen()) {
            CuDNN.CuDNNPtr filterBuffer = CuDNN.alloc(DirectImgBiasLayer.this.bias.length * Sizeof.DOUBLE);
            try {
              CuDNN.devicePool.with(device -> {
                CuDNN.handle(cudnnConvolutionBackwardBias(device.cudnnHandle, alpha.getPtr(),
                        inputDescriptor.getPtr(), errorPtr.getPtr(),
                        beta.getPtr(),
                        filterDescriptor.getPtr(), filterBuffer.getPtr()));
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error with " + Arrays.toString(inputSize),e);
            }
            final Tensor weightGradient = fromDevice(filterBuffer, new int[]{1,1,inputSize[2]});
            buffer.get(DirectImgBiasLayer.this, DirectImgBiasLayer.this.bias).accumulate(weightGradient.getData());
          }
          if (input.isAlive()) {
            input.accumulate(buffer, error);
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

  public double[] add(final double[] input) {
    //assert Arrays.stream(input).allMatch(v->Double.isFinite(v));
    assert(null != input);
    double[] bias = this.getBias();
    assert(null != bias);
    if(input.length % bias.length != 0) throw new IllegalArgumentException();
    final double[] array = new double[input.length];
    int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i/size];
    }
    assert Arrays.stream(array).allMatch(v->Double.isFinite(v));
    return array;
  }

  public DirectImgBiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getBias());
    return this;
  }

  public NNLayer set(final double[] ds) {
    double[] bias = this.getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    return this;
  }

  public DirectImgBiasLayer setWeights(final IntToDoubleFunction f) {
    double[] bias = this.getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getBias());
  }

  public double[] getBias() {
    if(!Arrays.stream(bias).allMatch(v->Double.isFinite(v))) {
      throw new RuntimeException(Arrays.toString(bias));
    }
    return bias;
  }
}
