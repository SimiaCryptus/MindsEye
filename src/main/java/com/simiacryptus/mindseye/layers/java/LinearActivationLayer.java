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
 * The type Linear activation layer.
 */
public class LinearActivationLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson());
    return json;
  }
  
  /**
   * From json linear activation layer.
   *
   * @param json the json
   * @return the linear activation layer
   */
  public static LinearActivationLayer fromJson(JsonObject json) {
    return new LinearActivationLayer(json);
  }
  
  /**
   * Instantiates a new Linear activation layer.
   *
   * @param json the json
   */
  protected LinearActivationLayer(JsonObject json) {
    super(json);
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  private final Tensor weights;
  
  /**
   * Instantiates a new Linear activation layer.
   */
  public LinearActivationLayer() {
    super();
    this.weights = new Tensor(2);
    this.weights.set(0, 1.);
    this.weights.set(1, 0.);
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    int itemCnt = inObj[0].getData().length();
    final double scale = this.weights.get(0);
    final double bias = this.weights.get(1);
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return input.map(v -> scale * v + bias);
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }
  
  /**
   * Gets bias.
   *
   * @return the bias
   */
  public double getBias() {
    return this.weights.get(1);
  }
  
  /**
   * Sets bias.
   *
   * @param bias the bias
   * @return the bias
   */
  public LinearActivationLayer setBias(double bias) {
    this.weights.set(1, bias);
    return this;
  }
  
  /**
   * Gets scale.
   *
   * @return the scale
   */
  public double getScale() {
    return this.weights.get(0);
  }
  
  /**
   * Sets scale.
   *
   * @param scale the scale
   * @return the scale
   */
  public LinearActivationLayer setScale(double scale) {
    this.weights.set(0, scale);
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
          final Tensor weightDelta = new Tensor(LinearActivationLayer.this.weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            weightDelta.add(0, deltaData[i] * inputData[inputData.length==1?0:i]);
            weightDelta.add(1, deltaData[i]);
          }
          buffer.get(LinearActivationLayer.this, LinearActivationLayer.this.weights).accumulate(weightDelta.getData());
        });
      }
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = this.inObj.getData().get(dataIndex).getDimensions();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i] * LinearActivationLayer.this.weights.getData()[0]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        this.inObj.accumulate(buffer, new TensorArray(passbackA));
        for(Tensor t : passbackA) t.release();
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }
}
