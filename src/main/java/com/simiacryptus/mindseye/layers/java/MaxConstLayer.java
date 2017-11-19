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
 * The type Max const layer.
 */
@SuppressWarnings("serial")
public class MaxConstLayer extends SimpleActivationLayer<MaxConstLayer> {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }
  
  /**
   * From json max const layer.
   *
   * @param json the json
   * @return the max const layer
   */
  public static MaxConstLayer fromJson(JsonObject json) {
    MaxConstLayer obj = new MaxConstLayer(json);
    obj.value = json.get("value").getAsDouble();
    return obj;
  }

  /**
   * Instantiates a new Max const layer.
   *
   * @param id the id
   */
  protected MaxConstLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Max const layer.
   */
  public MaxConstLayer() {
    super();
  }
  
  
  private double value = 0;
  
  @Override
  protected void eval(final double x, final double[] results) {
    final double d = x < this.value ? 0 : 1;
    final double f = x < this.value ? this.value : x;
    assert Double.isFinite(d);
    results[0] = f;
    results[1] = d;
  }
  
  /**
   * Gets value.
   *
   * @return the value
   */
  public double getValue() {
    return this.value;
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
