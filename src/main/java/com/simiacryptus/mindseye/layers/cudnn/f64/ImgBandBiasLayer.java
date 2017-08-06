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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Img band bias layer.
 */
public class ImgBandBiasLayer extends NNLayer {

  /**
   * From json img band bias layer.
   *
   * @param json the json
   * @return the img band bias layer
   */
  public static ImgBandBiasLayer fromJson(JsonObject json) {
    return new ImgBandBiasLayer(json);
  }

  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }

  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgBandBiasLayer(JsonObject json) {
    super(json);
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }

  private final double[] bias;

  /**
   * Instantiates a new Img band bias layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(int bands) {
    this.bias = new double[bands];
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }

  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    CuDNN.setDevice(nncontext.getCudaDeviceId());
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    assert(inputSize[2] == bias.length);
    int[] outputSize = inputSize;
    int length = batch.length();

    try {

      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnTensorDescriptor> filterDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
      CudaPtr alpha = CuDNN.javaPtr(nncontext.getCudaDeviceId(), 1.0);
      CudaPtr beta = CuDNN.javaPtr(nncontext.getCudaDeviceId(), 1.0);

      assert(0 < this.bias.length);
      CudaPtr filterPtr = CuDNN.write(nncontext.getCudaDeviceId(), this.bias);
      CudaPtr inputData = CudaPtr.toDeviceAsDouble(nncontext.getCudaDeviceId(), batch);
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
      TensorList output = CudaPtr.fromDeviceDouble(inputData, length, outputSize);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          CuDNN.setDevice(nncontext.getCudaDeviceId());
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          CudaPtr errorPtr = CudaPtr.toDeviceAsDouble(nncontext.getCudaDeviceId(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), ImgBandBiasLayer.this.bias.length * 1l * Sizeof.DOUBLE);
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
            final Tensor weightGradient = CudaPtr.fromDeviceDouble(filterBuffer, new int[]{1,1,inputSize[2]});
            //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
            DeltaBuffer deltaBuffer = buffer.get(ImgBandBiasLayer.this, ImgBandBiasLayer.this.bias);
            deltaBuffer.accumulate(weightGradient.getData());
            //assert Arrays.stream(deltaBuffer.delta).allMatch(Double::isFinite);
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

  /**
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  public double[] add(final double[] input) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(input).allMatch(v->Double.isFinite(v));
    assert(null != input);
    double[] bias = this.getBias();
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    assert(null != bias);
    if(input.length % bias.length != 0) throw new IllegalArgumentException();
    final double[] array = new double[input.length];
    int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i/size];
    }
    //assert Arrays.stream(array).allMatch(v->Double.isFinite(v));
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return array;
  }

  /**
   * Add weights img band bias layer.
   *
   * @param f the f
   * @return the img band bias layer
   */
  public ImgBandBiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getBias());
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return this;
  }

  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public NNLayer set(final double[] ds) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(ds).allMatch(Double::isFinite);
    double[] bias = this.getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return this;
  }

  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ImgBandBiasLayer setWeights(final IntToDoubleFunction f) {
    double[] bias = this.getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getBias());
  }

  /**
   * Get bias double [ ].
   *
   * @return the double [ ]
   */
  public double[] getBias() {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return bias;
  }
}
