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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.FastRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * The type Binary noise layer.
 */
@SuppressWarnings("serial")
public class BinaryNoiseLayer extends NNLayer implements StochasticComponent {
  
  
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
  private static final Logger log = LoggerFactory.getLogger(BinaryNoiseLayer.class);
  /**
   * The Mask list.
   */
  List<Tensor> maskList = new ArrayList<>();
  private double value;
  private boolean enabled = true;
  
  /**
   * Instantiates a new Binary noise layer.
   */
  public BinaryNoiseLayer() {
    this(0.5);
  }
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param value the value
   */
  public BinaryNoiseLayer(final double value) {
    super();
    setValue(value);
  }
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param json the json
   */
  protected BinaryNoiseLayer(final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
    JsonElement enabled = json.get("enabled");
    this.enabled = enabled == null || enabled.getAsBoolean();
  }
  
  /**
   * From json binary noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the binary noise layer
   */
  public static BinaryNoiseLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new BinaryNoiseLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    if (!enabled) return input;
    final int[] dimensions = input.getData().getDimensions();
    if (maskList.size() > 1 && !Arrays.equals(maskList.get(0).getDimensions(), dimensions)) {
      maskList.clear();
    }
    final int length = input.getData().length();
    final Tensor tensorPrototype = new Tensor(dimensions);
    while (length > maskList.size()) {
      maskList.add(tensorPrototype.map(v -> FastRandom.random() < getValue() ? 0 : (1.0 / getValue())));
    }
    final TensorArray mask = new TensorArray(maskList.stream().limit(length).toArray(i -> new Tensor[i]));
    return new NNResult(mask) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        input.accumulate(buffer, data);
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    json.addProperty("enabled", enabled);
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
  public BinaryNoiseLayer setValue(final double value) {
    this.value = value;
    shuffle();
    return this;
  }
  
  @Override
  public void shuffle() {
    maskList.clear();
  }
  
  @Override
  public void clearNoise() {
    maskList.clear();
    this.enabled = false;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
