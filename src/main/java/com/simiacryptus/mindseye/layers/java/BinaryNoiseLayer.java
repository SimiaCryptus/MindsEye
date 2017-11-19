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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.FastRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The type Binary noise layer.
 */
public class BinaryNoiseLayer extends NNLayer implements StochasticComponent {
  
  
  /**
   * The constant random.
   */
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BinaryNoiseLayer.class);
  private final int[] lastDim = {};
  private final int lastLength = 0;
  /**
   * The Mask list.
   */
  List<Tensor> maskList = new ArrayList<>();
  private double value;
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param json the json
   */
  protected BinaryNoiseLayer(JsonObject json) {
    super(json);
    this.value = json.get("value").getAsDouble();
  }
  
  /**
   * Instantiates a new Binary noise layer.
   *
   * @param value the value
   */
  public BinaryNoiseLayer(double value) {
    super();
    this.setValue(value);
  }
  
  /**
   * Instantiates a new Binary noise layer.
   */
  public BinaryNoiseLayer() {
    this(0.5);
  }
  
  /**
   * From json binary noise layer.
   *
   * @param json the json
   * @return the binary noise layer
   */
  public static BinaryNoiseLayer fromJson(JsonObject json) {
    return new BinaryNoiseLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
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
  public BinaryNoiseLayer setValue(double value) {
    this.value = value;
    shuffle();
    return this;
  }
  
  @Override
  public void shuffle() {
    maskList.clear();
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    int[] dimensions = input.getData().getDimensions();
    int length = input.getData().length();
    if (maskList.size() > 1 && !Arrays.equals(maskList.get(0).getDimensions(), dimensions)) maskList.clear();
    Tensor tensorPrototype = new Tensor(dimensions);
    while (length > maskList.size()) {
      maskList.add(tensorPrototype.map(v -> (FastRandom.random() < getValue()) ? 0 : 1));
    }
    TensorArray mask = new TensorArray(maskList.stream().limit(length).toArray(i -> new Tensor[i]));
    return new NNResult(mask) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        input.accumulate(buffer, new TensorArray());
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
