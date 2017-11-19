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
import java.util.stream.IntStream;

/**
 * The type Hyperbolic activation layer.
 */
public class HyperbolicActivationLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  private final Tensor weights;
  private int negativeMode = 1;
  
  /**
   * Instantiates a new Hyperbolic activation layer.
   *
   * @param json the json
   */
  protected HyperbolicActivationLayer(JsonObject json) {
    super(json);
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
    this.negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }
  
  /**
   * Instantiates a new Hyperbolic activation layer.
   */
  public HyperbolicActivationLayer() {
    super();
    this.weights = new Tensor(2);
    this.weights.set(0, 1.);
    this.weights.set(1, 1.);
  }
  
  /**
   * From json hyperbolic activation layer.
   *
   * @param json the json
   * @return the hyperbolic activation layer
   */
  public static HyperbolicActivationLayer fromJson(JsonObject json) {
    return new HyperbolicActivationLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson());
    json.addProperty("negativeMode", negativeMode);
    return json;
  }
  
  /**
   * Sets mode even.
   *
   * @return the mode even
   */
  public HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this;
  }
  
  /**
   * Sets mode odd.
   *
   * @return the mode odd
   */
  public HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this;
  }
  
  /**
   * Sets mode asymetric.
   *
   * @return the mode asymetric
   */
  public HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    int itemCnt = inObj[0].getData().length();
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return input.map(v -> {
        final int sign = v < 0 ? negativeMode : 1;
        final double a = Math.max(0, this.weights.get(v < 0 ? 1 : 0));
        return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
      });
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }
  
  /**
   * Gets scale r.
   *
   * @return the scale r
   */
  public double getScaleR() {
    return 1 / this.weights.get(0);
  }
  
  /**
   * Gets scale l.
   *
   * @return the scale l
   */
  public double getScaleL() {
    return 1 / this.weights.get(1);
  }
  
  /**
   * Sets scale.
   *
   * @param scale the scale
   * @return the scale
   */
  public HyperbolicActivationLayer setScale(double scale) {
    this.weights.set(0, 1 / scale);
    this.weights.set(1, 1 / scale);
    return this;
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
      
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).forEach(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final double[] inputData = this.inObj.getData().get(dataIndex).getData();
          final Tensor weightDelta = new Tensor(HyperbolicActivationLayer.this.weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            double d = deltaData[i];
            double x = inputData[i];
            final int sign = x < 0 ? negativeMode : 1;
            double a = Math.max(0, HyperbolicActivationLayer.this.weights.getData()[x < 0 ? 1 : 0]);
            weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
          }
          buffer.get(HyperbolicActivationLayer.this, HyperbolicActivationLayer.this.weights).accumulate(weightDelta.getData());
        });
      }
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = this.inObj.getData().get(dataIndex).getDimensions();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            double x = this.inObj.getData().get(dataIndex).getData()[i];
            double d = deltaData[i];
            final int sign = x < 0 ? negativeMode : 1;
            double a = Math.max(0, HyperbolicActivationLayer.this.weights.getData()[x < 0 ? 1 : 0]);
            passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x));
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
