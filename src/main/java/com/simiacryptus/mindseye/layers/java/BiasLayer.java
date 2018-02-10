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
import com.simiacryptus.mindseye.lang.*;
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
 * Adds a bias tensor to the input. Expects a single input of the same dimension as the bias tensor.
 */
@SuppressWarnings("serial")
public class BiasLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  /**
   * The Bias.
   */
  @Nullable
  public final double[] bias;
  
  /**
   * Instantiates a new Bias layer.
   */
  protected BiasLayer() {
    super();
    bias = null;
  }
  
  /**
   * Instantiates a new Bias layer.
   *
   * @param dims the dims
   */
  public BiasLayer(final int... dims) {
    bias = new double[Tensor.dim(dims)];
  }
  
  
  /**
   * Instantiates a new Bias layer.
   *
   * @param json the json
   */
  protected BiasLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  /**
   * From json bias layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the bias layer
   */
  public static BiasLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new BiasLayer(json);
  }
  
  /**
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  public double[] add(@javax.annotation.Nonnull final double[] input) {
    final double[] array = RecycleBin.DOUBLES.obtain(input.length);
    if (1 == bias.length) {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[0];
      }
    }
    else {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[i];
      }
    }
    return array;
  }
  
  /**
   * Add weights bias layer.
   *
   * @param f the f
   * @return the bias layer
   */
  @javax.annotation.Nonnull
  public BiasLayer addWeights(@javax.annotation.Nonnull final DoubleSupplier f) {
    Util.add(f, bias);
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    TensorList input;
    if (0 == inObj.length) {
      input = TensorArray.create();
    }
    else {
      input = inObj[0].getData();
    }
    return new NNResult(TensorArray.wrap(input.stream().parallel()
      .map(r -> {
        @Nonnull Tensor tensor = new Tensor(add(r.getData()), r.getDimensions());
        r.freeRef();
        return tensor;
      }).toArray(i -> new Tensor[i])),
      (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
        if (!isFrozen()) {
          final Delta<NNLayer> deltaBuffer = buffer.get(BiasLayer.this, bias);
          if (1 == bias.length) {
            data.stream().parallel().forEach(d -> {
              @Nullable final double[] array = d.getData();
              deltaBuffer.addInPlace(1 == array.length ? array : new double[]{Arrays.stream(array).sum()});
              d.freeRef();
            });
          }
          else {
            data.stream().parallel().forEach(d -> {
              deltaBuffer.addInPlace(d.getData()).freeRef();
              d.freeRef();
            });
          }
          deltaBuffer.freeRef();
        }
        if (0 < inObj.length && inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, data);
        }
      }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      
      @Override
      public boolean isAlive() {
        return 0 < inObj.length && inObj[0].isAlive() || !isFrozen();
      }
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(bias));
    return json;
  }
  
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  public NNLayer set(@javax.annotation.Nonnull final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @javax.annotation.Nonnull
  public BiasLayer setWeights(@javax.annotation.Nonnull final IntToDoubleFunction f) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  @javax.annotation.Nonnull
  public BiasLayer setWeightsLog(final double value) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.random() - 0.5) * Math.pow(10, value);
    }
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(bias);
  }
  
  /**
   * Set bias layer.
   *
   * @param tensor the tensor
   * @return the bias layer
   */
  @javax.annotation.Nonnull
  public BiasLayer set(@javax.annotation.Nonnull Tensor tensor) {
    assert bias.length == tensor.dim();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = tensor.get(i);
    }
    return this;
  }
}
