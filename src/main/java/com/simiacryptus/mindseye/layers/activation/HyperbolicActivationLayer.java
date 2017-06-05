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

package com.simiacryptus.mindseye.layers.activation;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;

public class HyperbolicActivationLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson());
    json.addProperty("negativeMode", negativeMode);
    return json;
  }
  
  public static HyperbolicActivationLayer fromJson(JsonObject json) {
    return new HyperbolicActivationLayer(json);
  }
  protected HyperbolicActivationLayer(JsonObject json) {
    super(json);
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
    this.negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  private final Tensor weights;
  private int negativeMode = 1;
  
  public HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this;
  }
  
  public HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this;
  }
  
  public HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this;
  }
  
  public HyperbolicActivationLayer() {
    super();
    this.weights = new Tensor(2);
    this.weights.set(0, 1.);
    this.weights.set(1, 1.);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      return input.map(v -> {
        final int sign = v<0?negativeMode:1;
        final double a = Math.max(0, this.weights.get(v<0?1:0));
        return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
      });
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }
  
  public double getScaleR() {
    return 1/this.weights.get(0);
  }
  public double getScaleL() {
    return 1/this.weights.get(1);
  }
  
  public HyperbolicActivationLayer setScale(double scale) {
    this.weights.set(0, 1/scale);
    this.weights.set(1, 1/scale);
    return this;
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
  
      if (!isFrozen()) {
        IntStream.range(0, delta.length).forEach(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          final Tensor weightDelta = new Tensor(HyperbolicActivationLayer.this.weights.getDims());
          for (int i = 0; i < deltaData.length; i++) {
            double d = deltaData[i];
            double x = inputData[i];
            final int sign = x<0?negativeMode:1;
            double a = Math.max(0, HyperbolicActivationLayer.this.weights.getData()[x<0?1:0]);
            weightDelta.add(x<0?1:0, -sign * d /(a*a*Math.sqrt(1+Math.pow(a*x,2))));
          }
          buffer.get(HyperbolicActivationLayer.this, HyperbolicActivationLayer.this.weights).accumulate(weightDelta.getData());
        });
      }
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length).mapToObj(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            double x = this.inObj.data[dataIndex].getData()[i];
            double d = deltaData[i];
            final int sign = x<0?negativeMode:1;
            double a = Math.max(0, HyperbolicActivationLayer.this.weights.getData()[x<0?1:0]);
            passback.set(i, sign * d * a * x / Math.sqrt(1+a * x*a * x));
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        this.inObj.accumulate(buffer, passbackA);
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }
}
