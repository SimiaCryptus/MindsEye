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
import org.jetbrains.annotations.Nullable;
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
public class ImgBandBiasLayer extends NNLayer implements MultiPrecision<ImgBandBiasLayer> {
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
  protected ImgBandBiasLayer(@javax.annotation.Nonnull final JsonObject json) {
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
  public static ImgBandBiasLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandBiasLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public NNLayer getCompatibilityLayer() {
    log.info("Using compatibility layer for " + this);
    return new NNLayer() {
      @javax.annotation.Nonnull
      com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer inner = this.as(com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer.class);
  
      @javax.annotation.Nonnull
      @Override
      public NNResult eval(NNResult... array) {
        @javax.annotation.Nonnull NNResult result = inner.eval(array);
        return new NNResult(result.getData(), (DeltaSet<NNLayer> buffer, TensorList data) -> {
          throw new IllegalStateException();
        }) {
      
          @Override
          protected void _free() {
        
          }
      
          @Override
          public boolean isAlive() {
            return false;
          }
        };
      }
  
      @javax.annotation.Nonnull
      @Override
      public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
        throw new IllegalStateException();
      }
  
      @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
  public double[] add(@javax.annotation.Nonnull final double[] input) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(input).allMatch(v->Double.isFinite(v));
    assert null != input;
    final double[] bias = getBias();
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    assert null != bias;
    if (input.length % bias.length != 0) throw new IllegalArgumentException();
    @javax.annotation.Nonnull final double[] array = new double[input.length];
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
  @javax.annotation.Nonnull
  public ImgBandBiasLayer addWeights(@javax.annotation.Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return this;
  }
  
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    Arrays.stream(inObj).forEach(x -> x.addRef());
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    @javax.annotation.Nonnull final int[] inputSize = batch.get(0).getDimensions();
    assert inputSize[2] == bias.length : inputSize[2] + " != " + bias.length;
    @javax.annotation.Nonnull final int[] outputSize = inputSize;
    final int length = batch.length();
  
    return new NNResult(GpuSystem.eval(nncontext -> {
      try {
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = GpuSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> filterDescriptor = GpuSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
  
        assert 0 < bias.length;
        @javax.annotation.Nonnull final CudaPtr filterPtr = CudaPtr.allocate(nncontext.getDeviceNumber(), (long) (bias.length * precision.size), MemoryType.Managed, false).write(precision, bias);
        final CudaPtr inputData = CudaPtr.getCudaPtr(precision, batch).asCopy();
        final @Nullable cudnnHandle cudnnHandle = nncontext.getHandle();
        try {
          GpuSystem.handle(CuDNNHandle.cudnnAddTensor(cudnnHandle, precision.getPointer(1.0),
                                                      filterDescriptor.getPtr(), filterPtr.getPtr(),
                                                      precision.getPointer(1.0),
                                                      inputDescriptor.getPtr(), inputData.getPtr()));
        } catch (@javax.annotation.Nonnull final Throwable e) {
          throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
        }
        filterPtr.freeRef();
        return GpuTensorList.wrap(inputData, length, outputSize, precision);
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
      }
    }), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList error) -> {
      assert error.length() == batch.length();
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      if (!isFrozen()) {
        GpuSystem.run(nncontext -> {
          final @Nullable cudnnHandle cudnnHandle = nncontext.getHandle();
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = GpuSystem.newTensorDescriptor(
            precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> filterDescriptor = GpuSystem.newTensorDescriptor(
            precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
          final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
          @javax.annotation.Nonnull final CudaPtr filterBuffer = CudaPtr.allocate(nncontext.getDeviceNumber(), bias.length * 1l * precision.size, MemoryType.Managed, false);
          try {
            GpuSystem.handle(CuDNNHandle.cudnnConvolutionBackwardBias(cudnnHandle, precision.getPointer(1.0),
                                                                      inputDescriptor.getPtr(), errorPtr.getPtr(),
                                                                      precision.getPointer(1.0),
                                                                      filterDescriptor.getPtr(), filterBuffer.getPtr()));
          } catch (@javax.annotation.Nonnull final Throwable e) {
            throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
          }
          errorPtr.freeRef();
          @javax.annotation.Nonnull final Tensor weightGradient = CudaPtr.read(filterBuffer, precision, new int[]{1, 1, inputSize[2]});
          filterBuffer.freeRef();
          //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
          final Delta<NNLayer> deltaBuffer = buffer.get(ImgBandBiasLayer.this, bias);
          deltaBuffer.addInPlace(weightGradient.getData());
          //assert Arrays.stream(deltaBuffer.delta).allMatch(Double::isFinite);
        });
      }
      if (input.isAlive()) {
        input.accumulate(buffer, error);
      }
    }) {
    
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
    
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
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
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
  public NNLayer set(@javax.annotation.Nonnull final double[] ds) {
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
  @javax.annotation.Nonnull
  public NNLayer set(@javax.annotation.Nonnull final Tensor ds) {
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
  @javax.annotation.Nonnull
  public ImgBandBiasLayer setWeights(@javax.annotation.Nonnull final IntToDoubleFunction f) {
    final double[] bias = getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    //assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
    return this;
  }
  
  @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
  public ImgBandBiasLayer setWeightsLog(int mag) {
    setWeights(i -> Math.pow(10, mag) * Math.random());
    return this;
  }
}
