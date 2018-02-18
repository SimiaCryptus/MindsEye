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
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * Adds a scalar offset to the input based on color band (3rd tensor dimension)
 */
@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase implements MultiPrecision<ImgBandBiasLayer> {
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
  public Layer getCompatibilityLayer() {
    log.info("Using compatibility layer for " + this);
    return new LayerBase() {
      @javax.annotation.Nonnull
      com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer inner = this.as(com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer.class);
  
      @javax.annotation.Nonnull
      @Override
      public Result eval(Result... array) {
        @javax.annotation.Nonnull Result result = inner.eval(array);
        return new Result(result.getData(), (DeltaSet<Layer> buffer, TensorList data) -> {
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
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    Arrays.stream(inObj).forEach(x -> x.addRef());
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @javax.annotation.Nonnull final int[] inputSize = batch.getDimensions();
    assert inputSize[2] == bias.length : inputSize[2] + " != " + bias.length;
    @javax.annotation.Nonnull final int[] outputSize = inputSize;
    final int length = batch.length();
  
    return new Result(CudaSystem.eval(gpu -> {
      try {
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = CudaSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> filterDescriptor = CudaSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
        
        assert 0 < bias.length;
        @javax.annotation.Nonnull final CudaPtr filterPtr = CudaPtr.allocate(gpu.getDeviceNumber(), (long) (bias.length * precision.size), MemoryType.Managed, true).write(precision, bias);
        final CudaPtr inputData = CudaPtr.getCudaPtr(precision, batch);
        try {
          CudaSystem.handle(gpu.cudnnAddTensor(precision.getPointer(1.0),
            filterDescriptor.getPtr(), filterPtr.getPtr(),
            precision.getPointer(1.0),
            inputDescriptor.getPtr(), inputData.getPtr()));
        } catch (@javax.annotation.Nonnull final Throwable e) {
          throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
        }
        gpu.registerForCleanup(inputDescriptor, filterPtr, filterDescriptor);
        return CudaTensorList.wrap(inputData, length, outputSize, precision);
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
      }
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList error) -> {
      assert error.length() == batch.length();
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      if (!isFrozen()) {
        CudaSystem.run(gpu -> {
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = CudaSystem.newTensorDescriptor(
            precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> filterDescriptor = CudaSystem.newTensorDescriptor(
            precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
          @Nullable final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
          @javax.annotation.Nonnull final CudaPtr filterBuffer = CudaPtr.allocate(gpu.getDeviceNumber(), bias.length * 1l * precision.size, MemoryType.Managed, false);
          try {
            try {
              CudaSystem.handle(gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0),
                inputDescriptor.getPtr(), errorPtr.getPtr(),
                precision.getPointer(1.0),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
            } catch (@javax.annotation.Nonnull final Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            @javax.annotation.Nonnull final Tensor weightGradient = CudaPtr.read(filterBuffer, precision, new int[]{1, 1, inputSize[2]});
            //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
            buffer.get(this, bias).addInPlace(weightGradient.getData()).freeRef();
            gpu.registerForCleanup(weightGradient);
          } finally {
            gpu.registerForCleanup(filterDescriptor, inputDescriptor, errorPtr, filterBuffer);
          }
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
  public Layer set(@javax.annotation.Nonnull final double[] ds) {
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
  public Layer set(@javax.annotation.Nonnull final Tensor ds) {
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
