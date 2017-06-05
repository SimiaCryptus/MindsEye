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
import com.simiacryptus.mindseye.layers.meta.CrossDotMetaLayer;

import java.util.UUID;

public final class SqActivationLayer extends SimpleActivationLayer<SqActivationLayer> {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static SqActivationLayer fromJson(JsonObject json) {
    return new SqActivationLayer(json);
  }
  protected SqActivationLayer(JsonObject id) {
    super(id);
  }
  
  private static final long serialVersionUID = -5520500379591109767L;
  
  public SqActivationLayer() {
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = 2 * x;
    final double f = x * x;
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
  
}
