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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.stochastic.BinaryNoiseLayer;
import com.simiacryptus.mindseye.layers.stochastic.StochasticComponent;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The type Dropout noise layer.
 */
public class DropoutNoiseLayer extends PipelineNetwork implements StochasticComponent {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJson();
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
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DropoutNoiseLayer.class);
  
  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param value the value
   */
  public DropoutNoiseLayer(double value) {
    super(1);
    add(new ProductInputsLayer(),
      add(new BinaryNoiseLayer().setName("mask"), getInput(0)),
      getInput(0));
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
    return this.<BinaryNoiseLayer>getByName("mask").getValue();
  }
  
  /**
   * Sets value.
   *
   * @param value the value
   * @return the value
   */
  public DropoutNoiseLayer setValue(double value) {
    this.<BinaryNoiseLayer>getByName("mask").setValue(value);
    return this;
  }
  
  /**
   * Shuffle.
   */
  @Override
  public void shuffle() {
    visit(layer->{
      if(layer instanceof StochasticComponent) ((StochasticComponent)layer).shuffle();
    });
  }
  
}
