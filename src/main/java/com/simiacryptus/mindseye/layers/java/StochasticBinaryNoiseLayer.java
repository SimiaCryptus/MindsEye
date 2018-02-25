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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The type Binary noise layer.
 */
@SuppressWarnings("serial")
public class StochasticBinaryNoiseLayer extends LayerBase implements StochasticComponent {
  
  
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
  private static final Logger log = LoggerFactory.getLogger(StochasticBinaryNoiseLayer.class);
  private final double amplitude;
  /**
   * The Mask list.
   */
  @Nullable
  volatile Tensor mask = null;
  @Nonnull
  int[] dimensions;
  private double density;
  private long seed = System.nanoTime();
  private long layerSeed = System.nanoTime();
  
  /**
   * Instantiates a new Binary noise layer.
   */
  public StochasticBinaryNoiseLayer() {this(new int[]{});}
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param dimensions
   */
  public StochasticBinaryNoiseLayer(final int... dimensions) {
    this(0.5, 1.0, dimensions);
  }
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param density    the value
   * @param amplitude
   * @param dimensions
   */
  public StochasticBinaryNoiseLayer(final double density, final double amplitude, final int... dimensions) {
    super();
    setDensity(density);
    this.amplitude = amplitude;
    this.dimensions = dimensions;
  }
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param json the json
   */
  protected StochasticBinaryNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    density = json.get("density").getAsDouble();
    amplitude = json.get("amplitude").getAsDouble();
    dimensions = Tensor.fromJsonArray(json.get("dimensions").getAsJsonArray());
    seed = json.get("seed").getAsLong();
    layerSeed = json.get("layerSeed").getAsLong();
  }
  
  /**
   * From json binary noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the binary noise layer
   */
  public static StochasticBinaryNoiseLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new StochasticBinaryNoiseLayer(json);
  }
  
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert null == inObj || 0 == inObj.length;
    if (null == mask) {
      synchronized (this) {
        if (null == mask) {
          mask = new Tensor(dimensions);
          Random random = new Random(seed ^ layerSeed);
          for (int i = 0; i < mask.length(); i++) {
            mask.set(i, random.nextBoolean() ? amplitude : 0);
          }
        }
      }
    }
    return new Result(TensorArray.create(mask), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {});
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("density", density);
    json.addProperty("amplitude", amplitude);
    json.addProperty("seed", seed);
    json.addProperty("layerSeed", layerSeed);
    json.add("dimensions", Tensor.toJsonArray(dimensions));
    return json;
  }
  
  /**
   * Gets value.
   *
   * @return the value
   */
  public double getDensity() {
    return density;
  }
  
  /**
   * Sets value.
   *
   * @param density the value
   * @return the value
   */
  @Nonnull
  public StochasticBinaryNoiseLayer setDensity(final double density) {
    this.density = density;
    shuffle(StochasticComponent.random.get().nextLong());
    return this;
  }
  
  @Override
  public void shuffle(final long seed) {
    this.seed = seed;
    mask = null;
  }
  
  @Override
  public void clearNoise() {
    if (null != mask) mask.freeRef();
    mask = null;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
