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
 * Enforces a maximum-value constraint on the input signal, rounding down any values exceeding a setByCoord threshold.
 */
@SuppressWarnings("serial")
public class MaxConstLayer extends SimpleActivationLayer<MaxConstLayer> {
  
  private double value = 0;
  
  /**
   * Instantiates a new Max const layer.
   */
  public MaxConstLayer() {
    super();
  }
  
  /**
   * Instantiates a new Max const layer.
   *
   * @param id the id
   */
  protected MaxConstLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json max const layer.
   *
   * @param json the json
   * @return the max const layer
   */
  public static MaxConstLayer fromJson(final JsonObject json) {
    final MaxConstLayer obj = new MaxConstLayer(json);
    obj.value = json.get("value").getAsDouble();
    return obj;
  }
  
  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x < value ? 0 : 1;
    final double f = x < value ? value : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }
  
  @Override
  public JsonObject getJson() {
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
  public MaxConstLayer setValue(final double value) {
    this.value = value;
    return this;
  }
}
