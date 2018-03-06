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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * Adds a per-color-band value offset to the single tensor input.
 */
@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  @Nullable
  private final double[] bias;
  
  /**
   * Instantiates a new Img band bias layer.
   */
  protected ImgBandBiasLayer() {
    super();
    bias = null;
  }
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(final int bands) {
    super();
    bias = new double[bands];
  }
  
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgBandBiasLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
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
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  @javax.annotation.Nonnull
  public double[] add(@javax.annotation.Nonnull final double[] input) {
    assert Arrays.stream(input).allMatch(v -> Double.isFinite(v));
    assert null != input;
    @Nullable final double[] bias = getBias();
    assert null != bias;
    if (input.length % bias.length != 0) throw new IllegalArgumentException();
    @javax.annotation.Nonnull final double[] array = new double[input.length];
    final int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + bias[i / size];
    }
    assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
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
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Result eval(final Result... inObj) {
    return eval(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  @javax.annotation.Nonnull
  public Result eval(@javax.annotation.Nonnull final Result input) {
    @Nullable final double[] bias = getBias();
    input.addRef();
    return new Result(TensorArray.wrap(input.getData().stream().parallel()
      .map(r -> {
        if (r.getDimensions().length != 3) {
          throw new IllegalArgumentException(Arrays.toString(r.getDimensions()));
        }
        if (r.getDimensions()[2] != bias.length) {
          throw new IllegalArgumentException(String.format("%s: %s does not have %s bands",
            getName(), Arrays.toString(r.getDimensions()), bias.length));
        }
        @Nonnull Tensor tensor = new Tensor(add(r.getData()), r.getDimensions());
        r.freeRef();
        return tensor;
      })
      .toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (!isFrozen()) {
        final Delta<Layer> deltaBuffer = buffer.get(ImgBandBiasLayer.this, bias);
        data.stream().parallel().forEach(d -> {
          final double[] array = RecycleBin.DOUBLES.obtain(bias.length);
          @Nullable final double[] signal = d.getData();
          final int size = signal.length / bias.length;
          for (int i = 0; i < signal.length; i++) {
            array[i / size] += signal[i];
            if (!Double.isFinite(array[i / size])) {
              array[i / size] = 0.0;
            }
          }
          d.freeRef();
          assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
          deltaBuffer.addInPlace(array);
          RecycleBin.DOUBLES.recycle(array, array.length);
        });
        deltaBuffer.freeRef();
      }
      if (input.isAlive()) {
        data.addRef();
        input.accumulate(buffer, data);
      }
    }) {
  
      @Override
      protected void _free() {
        input.freeRef();
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
  @Nullable
  public double[] getBias() {
    if (!Arrays.stream(bias).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(Arrays.toString(bias));
    }
    return bias;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getBias()));
    return json;
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  public Layer set(@javax.annotation.Nonnull final double[] ds) {
    @Nullable final double[] bias = getBias();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
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
    @Nullable final double[] bias = getBias();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
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
   * @param value the value
   * @return the weights log
   */
  @javax.annotation.Nonnull
  public ImgBandBiasLayer setWeightsLog(final double value) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
    return this;
  }
  
  
  /**
   * Sets and free.
   *
   * @param tensor the tensor
   * @return the and free
   */
  public ImgBandBiasLayer setAndFree(final com.simiacryptus.mindseye.lang.Tensor tensor) {
    set(tensor.getData());
    tensor.freeRef();
    return this;
  }
  
  /**
   * Set img band bias layer.
   *
   * @param tensor the tensor
   * @return the img band bias layer
   */
  public ImgBandBiasLayer set(final com.simiacryptus.mindseye.lang.Tensor tensor) {
    return (ImgBandBiasLayer) set(tensor.getData());
  }
}
