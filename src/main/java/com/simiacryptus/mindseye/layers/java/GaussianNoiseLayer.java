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
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Adds uniform random gaussian noise to all input elements.
 */
@SuppressWarnings("serial")
public class GaussianNoiseLayer extends NNLayer {
  
  
  /**
   * The constant randomize.
   */
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(GaussianNoiseLayer.class);
  private long seed = GaussianNoiseLayer.random.get().nextLong();
  private double value;
  
  /**
   * Instantiates a new Gaussian noise layer.
   */
  public GaussianNoiseLayer() {
    super();
    setValue(1.0);
  }
  
  /**
   * Instantiates a new Gaussian noise layer.
   *
   * @param json the json
   */
  protected GaussianNoiseLayer(final @NotNull JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
  }
  
  /**
   * From json gaussian noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the gaussian noise layer
   */
  public static GaussianNoiseLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new GaussianNoiseLayer(json);
  }
  
  @Override
  public @NotNull NNResult eval(final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    inObj[0].addRef();
    inObj[0].getData().addRef();
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final @NotNull Random random = new Random(seed);
      final Tensor input = inObj[0].getData().get(dataIndex);
      final @Nullable Tensor output = input.map(x -> {
        return x + random.nextGaussian() * getValue();
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(TensorArray.wrap(outputA), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      if (inObj[0].isAlive()) {
        @NotNull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final @Nullable double[] deltaData = delta.get(dataIndex).getData();
          final @NotNull int[] dims = inObj[0].getData().get(dataIndex).getDimensions();
          final @NotNull Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
  
      @Override
      protected void _free() {
        inObj[0].freeRef();
      }
  
  
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }
  
  /**
   * Gets value.
   *
   * @return the value
   */
  public double getValue() {
    return value;
  }
  
  /**
   * Sets value.
   *
   * @param value the value
   * @return the value
   */
  public @NotNull GaussianNoiseLayer setValue(final double value) {
    this.value = value;
    return this;
  }
  
  /**
   * Shuffle.
   */
  public void shuffle() {
    seed = GaussianNoiseLayer.random.get().nextLong();
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
  
}
