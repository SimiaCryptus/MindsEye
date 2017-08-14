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
import java.util.stream.IntStream;

/**
 * The type Dropout noise layer.
 */
public class DropoutNoiseLayer extends NNLayer implements StochasticComponent {
  
  
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
  public static DropoutNoiseLayer fromJson(JsonObject json) {
    return new DropoutNoiseLayer(json);
  }

  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param json the json
   */
  protected DropoutNoiseLayer(JsonObject json) {
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
  private static final Logger log = LoggerFactory.getLogger(DropoutNoiseLayer.class);
  
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
  public DropoutNoiseLayer(double value) {
    super();
    this.setValue(value);
  }
  
  /**
   * Instantiates a new Dropout noise layer.
   */
  public DropoutNoiseLayer() {
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
  public DropoutNoiseLayer setValue(double value) {
    this.value = value;
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
    int itemCnt = inObj[0].getData().length();
    Random random = new Random(seed);
    Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = input.map(x -> {
        return (random.nextDouble() < getValue()) ? 0 : 1;
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final double[] input = inObj[0].getData().get(dataIndex).getData();
      final double[] maskT = mask[dataIndex].getData();
      final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0], mask);
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    private final Tensor[] mask;
    
    private Result(final Tensor[] outputA, final NNResult inObj, Tensor[] mask) {
      super(outputA);
      this.inObj = inObj;
      this.mask = mask;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = this.inObj.getData().get(dataIndex).getDimensions();
          double[] maskData = mask[dataIndex].getData();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        this.inObj.accumulate(buffer, new TensorArray(passbackA));
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }
  
}
