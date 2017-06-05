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

public final class NthPowerActivationLayer extends SimpleActivationLayer<NthPowerActivationLayer> {
  
  private double power = 1.0;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("power", power);
    return json;
  }
  public static NthPowerActivationLayer fromJson(JsonObject json) {
    return new NthPowerActivationLayer(json);
  }
  protected NthPowerActivationLayer(JsonObject id) {
    super(id);
    power = id.get("power").getAsDouble();
  }
  
  public NthPowerActivationLayer() {
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = power * Math.pow(x, power-1);
    final double f = Math.pow(x, power);
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
  
  public double getPower() {
    return power;
  }
  
  public NthPowerActivationLayer setPower(double power) {
    this.power = power;
    return this;
  }
}
