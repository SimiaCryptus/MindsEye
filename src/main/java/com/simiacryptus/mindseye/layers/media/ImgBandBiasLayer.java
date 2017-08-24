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

package com.simiacryptus.mindseye.layers.media;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * The type Img band bias layer.
 */
public class ImgBandBiasLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }
  
  /**
   * From json img band bias layer.
   *
   * @param json the json
   * @return the img band bias layer
   */
  public static ImgBandBiasLayer fromJson(JsonObject json) {
    return new ImgBandBiasLayer(json);
  }
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgBandBiasLayer(JsonObject json) {
    super(json);
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  
  
  private final double[] bias;
  
  /**
   * Instantiates a new Img band bias layer.
   */
  protected ImgBandBiasLayer() {
    super();
    this.bias = null;
  }
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(final int bands) {
    super();
    this.bias = new double[bands];
  }
  
  /**
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  public double[] add(final double[] input) {
    assert Arrays.stream(input).allMatch(v->Double.isFinite(v));
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
  
  /**
   * Add weights img band bias layer.
   *
   * @param f the f
   * @return the img band bias layer
   */
  public ImgBandBiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getBias());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    return eval(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  public NNResult eval(NNResult input) {
    final double[] bias = getBias();
    assert input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    Tensor[] outputA = input.getData().stream().parallel()
                           .map(r -> {
                             if(r.getDimensions().length != 3) throw new IllegalArgumentException(Arrays.toString(r.getDimensions()));
                             if(r.getDimensions()[2] != bias.length) throw new IllegalArgumentException(String.format("%s: %s does not have %s bands",
                                 getName(), Arrays.toString(r.getDimensions()), bias.length));
                             return new Tensor(r.getDimensions(), add(r.getData()));
                           })
                           .toArray(i -> new Tensor[i]);
    assert Arrays.stream(outputA).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        assert data.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        if (!isFrozen()) {
          Delta deltaBuffer = buffer.get(ImgBandBiasLayer.this, bias);
          data.stream().parallel().forEach(d -> {
            final double[] array = Tensor.obtain(bias.length);
            double[] signal = d.getData();
            int size = signal.length / bias.length;
            for (int i = 0; i < signal.length; i++) {
              array[i/size] += signal[i];
              if(!Double.isFinite(array[i/size])) array[i/size] = 0.0;
            }
            assert Arrays.stream(array).allMatch(v->Double.isFinite(v));
            deltaBuffer.accumulate(array);
            Tensor.recycle(array);
          });
        }
        if (input.isAlive()) {
          input.accumulate(buffer, data);
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public NNLayer set(final double[] ds) {
    double[] bias = this.getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
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
    assert Arrays.stream(bias).allMatch(v->Double.isFinite(v));
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
    if(!Arrays.stream(bias).allMatch(v->Double.isFinite(v))) {
      throw new RuntimeException(Arrays.toString(bias));
    }
    return bias;
  }
}
