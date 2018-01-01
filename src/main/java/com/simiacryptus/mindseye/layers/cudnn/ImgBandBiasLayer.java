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
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * Adds a scalar offset to the input based on color band (3rd tensor dimension)
 */
@SuppressWarnings("serial")
public class ImgBandBiasLayer extends NNLayer implements LayerPrecision<ImgBandBiasLayer> {
  
  private final double[] bias;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(final int bands) {
    bias = new double[bands];
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgBandBiasLayer(final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
    precision = Precision.valueOf(json.get("precision").getAsString());
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }
  
  /**
   * From json img band bias layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band bias layer
   */
  public static ImgBandBiasLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandBiasLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
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
    assert null != input;
    final double[] bias = getBias();
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    assert null != bias;
    if (input.length % bias.length != 0) throw new IllegalArgumentException();
    final double[] array = new double[input.length];
    final int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i / size];
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
    Util.add(f, getBias());
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return this;
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    if (((CudaExecutionContext) nncontext).getDeviceNumber() < 0) return getCompatibilityLayer().eval(nncontext, inObj);
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    assert inputSize[2] == bias.length : inputSize[2] + " != " + bias.length;
    final int[] outputSize = inputSize;
    final int length = batch.length();
    
    try {
  
      final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      final CudaResource<cudnnTensorDescriptor> filterDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
      final CudaPtr alpha = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      final CudaPtr beta = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
  
      assert 0 < bias.length;
      final CudaPtr filterPtr = new CudaPtr(bias.length * precision.size, ((CudaExecutionContext) nncontext).getDeviceNumber()).write(precision, bias);
      final CudaPtr inputData = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, batch);
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      try {
        CuDNN.handle(CuDNN.cudnnAddTensor(cudnnHandle, alpha.getPtr(),
                                          filterDescriptor.getPtr(), filterPtr.getPtr(),
                                          beta.getPtr(),
                                          inputDescriptor.getPtr(), inputData.getPtr()));
      } catch (final Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
      }
      final TensorList output = new GpuTensorList(inputData, length, outputSize, cudnnHandle, precision);
      return new NNResult(output) {
  
        @Override
        public void finalize() {
          Arrays.stream(inObj).forEach(NNResult::finalize);
        }
  
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert error.length() == batch.length();
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          if (!isFrozen()) {
            final CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), bias.length * 1l * precision.size);
            try {
              CuDNN.handle(CuDNN.cudnnConvolutionBackwardBias(cudnnHandle, alpha.getPtr(),
                                                              inputDescriptor.getPtr(), errorPtr.getPtr(),
                                                              beta.getPtr(),
                                                              filterDescriptor.getPtr(), filterBuffer.getPtr()));
            } catch (final Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            final Tensor weightGradient = CudaPtr.read(filterBuffer, precision, new int[]{1, 1, inputSize[2]});
            //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
            final Delta<NNLayer> deltaBuffer = buffer.get(ImgBandBiasLayer.this, bias);
            deltaBuffer.addInPlace(weightGradient.getData());
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
    } catch (final Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
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
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ImgBandBiasLayer setPrecision(final Precision precision) {
    this.precision = precision;
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
    final double[] bias = getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return this;
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public NNLayer set(final Tensor ds) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(ds).allMatch(Double::isFinite);
    final double[] bias = getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = ds.get(i);
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
    final double[] bias = getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getBias());
  }
}
