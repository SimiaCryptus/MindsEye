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
 * The type Entropy layer.
 */
@SuppressWarnings("serial")
public class EntropyLayer extends SimpleActivationLayer<EntropyLayer> {
  
  /**
   * Instantiates a new Entropy layer.
   *
   * @param id the id
   */
  protected EntropyLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Entropy layer.
   */
  public EntropyLayer() {
    super();
  }
  
  /**
   * From json entropy layer.
   *
   * @param json the json
   * @return the entropy layer
   */
  public static EntropyLayer fromJson(JsonObject json) {
    return new EntropyLayer(json);
  }
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  @Override
  protected void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    double d;
    double f;
    if (0. == x) {
      d = 0;
      f = 0;
    }
    else {
      final double log = Math.log(Math.abs(x));
      d = -(1 + log);
      f = -x * log;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = d;
  }
}
