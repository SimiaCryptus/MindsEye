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
 * The type Binary entropy activation layer.
 */
public final class BinaryEntropyActivationLayer extends SimpleActivationLayer<BinaryEntropyActivationLayer> {
  
  /**
   * Instantiates a new Binary entropy activation layer.
   *
   * @param id the id
   */
  protected BinaryEntropyActivationLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Binary entropy activation layer.
   */
  public BinaryEntropyActivationLayer() {
  }
  
  /**
   * From json binary entropy activation layer.
   *
   * @param json the json
   * @return the binary entropy activation layer
   */
  public static BinaryEntropyActivationLayer fromJson(JsonObject json) {
    return new BinaryEntropyActivationLayer(json);
  }
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double d = 0 == x ? Double.NaN : (Math.log(x) - Math.log(1 - x));
    final double f = (0 >= x || 1 <= x) ? Double.POSITIVE_INFINITY : (x * Math.log(x) + (1 - x) * Math.log(1 - x));
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
  
}
