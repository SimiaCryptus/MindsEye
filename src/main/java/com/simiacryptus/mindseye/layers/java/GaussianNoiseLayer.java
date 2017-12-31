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
  protected GaussianNoiseLayer(final JsonObject json) {
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
  public static GaussianNoiseLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new GaussianNoiseLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Random random = new Random(seed);
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = input.map(x -> {
        return x + random.nextGaussian() * getValue();
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
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
  public GaussianNoiseLayer setValue(final double value) {
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
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    @Override
    public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
      if (inObj.isAlive()) {
        final Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = inObj.getData().get(dataIndex).getDimensions();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        inObj.accumulate(buffer, new TensorArray(passbackA));
      }
    }
    
    @Override
    public boolean isAlive() {
      return inObj.isAlive() || !isFrozen();
    }
    
  }
}
