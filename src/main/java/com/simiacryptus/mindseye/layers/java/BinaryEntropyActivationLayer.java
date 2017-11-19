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
 * The type Log activation layer.
 */
public final class BinaryEntropyActivationLayer extends SimpleActivationLayer<BinaryEntropyActivationLayer> {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json log activation layer.
   *
   * @param json the json
   * @return the log activation layer
   */
  public static BinaryEntropyActivationLayer fromJson(JsonObject json) {
    return new BinaryEntropyActivationLayer(json);
  }

  /**
   * Instantiates a new Log activation layer.
   *
   * @param id the id
   */
  protected BinaryEntropyActivationLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Log activation layer.
   */
  public BinaryEntropyActivationLayer() {
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = 0 == x ? Double.NaN : (Math.log(x) - Math.log(1-x));
    final double f = (0 >= x || 1 <= x) ? Double.POSITIVE_INFINITY : (x * Math.log(x) + (1-x) * Math.log(1-x));
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
  
}
