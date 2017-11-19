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

/**
 * The type Sinewave activation layer.
 */
public final class SinewaveActivationLayer extends SimpleActivationLayer<SinewaveActivationLayer> {
  
  private boolean balanced = true;
  
  /**
   * Instantiates a new Sinewave activation layer.
   *
   * @param id the id
   */
  protected SinewaveActivationLayer(JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }
  
  /**
   * Instantiates a new Sinewave activation layer.
   */
  public SinewaveActivationLayer() {
  }
  
  /**
   * From json sinewave activation layer.
   *
   * @param json the json
   * @return the sinewave activation layer
   */
  public static SinewaveActivationLayer fromJson(JsonObject json) {
    return new SinewaveActivationLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
    return json;
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    double d = Math.cos(x);
    double f = Math.sin(x);
    if (!isBalanced()) {
      d = d / 2;
      f = (f + 1) / 2;
    }
    results[0] = f;
    results[1] = d;
  }
  
  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return this.balanced;
  }
  
  /**
   * Sets balanced.
   *
   * @param balanced the balanced
   * @return the balanced
   */
  public SinewaveActivationLayer setBalanced(final boolean balanced) {
    this.balanced = balanced;
    return this;
  }
}
