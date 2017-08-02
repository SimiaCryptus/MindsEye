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

import java.util.UUID;

/**
 * The type Nth power activation layer.
 */
public final class NthPowerActivationLayer extends SimpleActivationLayer<NthPowerActivationLayer> {
  
  private double power = 1.0;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("power", power);
    return json;
  }

  /**
   * From json nth power activation layer.
   *
   * @param json the json
   * @return the nth power activation layer
   */
  public static NthPowerActivationLayer fromJson(JsonObject json) {
    return new NthPowerActivationLayer(json);
  }

  /**
   * Instantiates a new Nth power activation layer.
   *
   * @param id the id
   */
  protected NthPowerActivationLayer(JsonObject id) {
    super(id);
    power = id.get("power").getAsDouble();
  }
  
  /**
   * Instantiates a new Nth power activation layer.
   */
  public NthPowerActivationLayer() {
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    assert(0 < results.length);
    assert Double.isFinite(x);
    boolean isZero = Math.abs(x) < 1e-20;
    double d = isZero?0.0:(power * Math.pow(x, power-1));
    double f = isZero?0.0:Math.pow(x, power);
    if(!Double.isFinite(d)) d = 0.0;
    if(!Double.isFinite(f)) f = 0.0;
    results[0] = f;
    results[1] = d;
  }
  
  /**
   * Gets power.
   *
   * @return the power
   */
  public double getPower() {
    return power;
  }
  
  /**
   * Sets power.
   *
   * @param power the power
   * @return the power
   */
  public NthPowerActivationLayer setPower(double power) {
    this.power = power;
    return this;
  }
}
