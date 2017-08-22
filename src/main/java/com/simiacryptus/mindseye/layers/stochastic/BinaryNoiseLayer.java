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

package com.simiacryptus.mindseye.layers.stochastic;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The type Dropout noise layer.
 */
public class BinaryNoiseLayer extends NNLayer implements StochasticComponent {
  
  
  private TensorArray mask = null;
  private int[] lastDim = new int[]{};
  private int lastLength = 0;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }
  
  /**
   * From json dropout noise layer.
   *
   * @param json the json
   * @return the dropout noise layer
   */
  public static BinaryNoiseLayer fromJson(JsonObject json) {
    return new BinaryNoiseLayer(json);
  }

  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param json the json
   */
  protected BinaryNoiseLayer(JsonObject json) {
    super(json);
    this.value = json.get("value").getAsDouble();
  }
  
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
  
  /**
   * The Seed.
   */
  long seed = random.get().nextLong();
  private double value;
  
  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param value the value
   */
  public BinaryNoiseLayer(double value) {
    super();
    this.setValue(value);
  }
  
  /**
   * Instantiates a new Dropout noise layer.
   */
  public BinaryNoiseLayer() {
    this(0.5);
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
  
  /**
   * Shuffle.
   */
  @Override
  public void shuffle() {
    seed = random.get().nextLong();
  }
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    int[] dimensions = input.getData().getDimensions();
    int length = input.getData().length();
    if(length != lastLength || !Arrays.equals(dimensions, lastDim)) this.mask = null;
    Random random = new Random(seed);
    if(null == this.mask) this.mask = new TensorArray(input.getData().stream().map(x -> x.map(v -> (random.nextDouble() < getValue()) ? 0 : 1)).toArray(i -> new Tensor[i]));
    this.lastLength = length;
    this.lastDim = dimensions;
    return new NNResult(mask) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        input.accumulate(buffer, new TensorArray(data.stream().map(x->x.map(v->0.0))
                                                   .toArray(i->new Tensor[i])));
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
