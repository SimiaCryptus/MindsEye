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
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  
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
    log.info("Using compatibility layer for " + this);
    return new NNLayer() {
      com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer inner = this.as(com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer.class);
    
      @Override
      public NNResult eval(NNResult... array) {
        NNResult result = inner.eval(array);
        return new NNResult(result.getData()) {
          @Override
          public void accumulate(DeltaSet<NNLayer> buffer, TensorList data) {
            throw new IllegalStateException();
          }
        
          @Override
          public boolean isAlive() {
            return false;
          }
        };
      }
    
      @Override
      public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
        throw new IllegalStateException();
      }
    
      @Override
      public List<double[]> state() {
        throw new IllegalStateException();
      }
    };
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
  public NNResult eval(final NNResult... inObj) {
    if (!CuDNN.isEnabled()) return getCompatibilityLayer().eval(inObj);
    return GpuHandle.run(nncontext -> {
      if (nncontext.getDeviceNumber() < 0) return getCompatibilityLayer().eval(inObj);
      nncontext.initThread();
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
  
        assert 0 < bias.length;
        final CudaPtr filterPtr = CudaPtr.allocate(nncontext.getDeviceNumber(), (long) (bias.length * precision.size), MemoryType.Managed, false).write(precision, bias);
        final CudaPtr inputData = CudaPtr.getCudaPtr(precision, batch);
        final cudnnHandle cudnnHandle = nncontext.getHandle();
        try {
          CuDNN.handle(CuDNN.cudnnAddTensor(cudnnHandle, precision.getPointer(1.0),
                                            filterDescriptor.getPtr(), filterPtr.getPtr(),
                                            precision.getPointer(1.0),
                                            inputDescriptor.getPtr(), inputData.getPtr()));
        } catch (final Throwable e) {
          throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
        }
        final TensorList output = GpuTensorList.create(inputData, length, outputSize, precision);
        return new NNResult(output) {
        
          @Override
          public void free() {
            Arrays.stream(inObj).forEach(NNResult::free);
          }
        
          @Override
          public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
            assert error.length() == batch.length();
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
            if (!isFrozen()) {
              GpuHandle.apply(nncontext -> {
                final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
                final CudaPtr filterBuffer = CudaPtr.allocate(nncontext.getDeviceNumber(), bias.length * 1l * precision.size, MemoryType.Managed, false);
                try {
                  CuDNN.handle(CuDNN.cudnnConvolutionBackwardBias(cudnnHandle, precision.getPointer(1.0),
                                                                  inputDescriptor.getPtr(), errorPtr.getPtr(),
                                                                  precision.getPointer(1.0),
                                                                  filterDescriptor.getPtr(), filterBuffer.getPtr()));
                } catch (final Throwable e) {
                  throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
                }
                final Tensor weightGradient = CudaPtr.read(filterBuffer, precision, new int[]{1, 1, inputSize[2]});
                //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
                final Delta<NNLayer> deltaBuffer = buffer.get(ImgBandBiasLayer.this, bias);
                deltaBuffer.addInPlace(weightGradient.getData());
                //assert Arrays.stream(deltaBuffer.delta).allMatch(Double::isFinite);
              });
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
    });
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
  
  /**
   * Sets weights log.
   *
   * @param mag the mag
   * @return the weights log
   */
  public ImgBandBiasLayer setWeightsLog(int mag) {
    setWeights(i -> Math.pow(10, mag) * Math.random());
    return this;
  }
}
