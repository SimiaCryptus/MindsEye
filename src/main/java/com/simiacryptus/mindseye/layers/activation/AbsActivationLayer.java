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

public final class AbsActivationLayer extends SimpleActivationLayer<AbsActivationLayer> {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static AbsActivationLayer fromJson(JsonObject json) {
    return new AbsActivationLayer(json);
  }
  protected AbsActivationLayer(JsonObject id) {
    super(id);
  }
  
  private static final long serialVersionUID = -5520500379591109767L;
  
  public AbsActivationLayer() {
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = x < 0 ? -1 : 1;
    final double f = x < 0 ? -x : x;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
  
}
