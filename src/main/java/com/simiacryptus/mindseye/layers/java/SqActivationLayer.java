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
import com.simiacryptus.mindseye.lang.DataSerializer;

import java.util.Map;

/**
 * Specialized square activation function. Deprecated. Use NthPowerActivationLayer.
 */
@SuppressWarnings("serial")
public final class SqActivationLayer extends SimpleActivationLayer<SqActivationLayer> {
  
  /**
   * Instantiates a new Sq activation layer.
   */
  public SqActivationLayer() {
  }
  
  /**
   * Instantiates a new Sq activation layer.
   *
   * @param id the id
   */
  protected SqActivationLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sq activation layer.
   *
   * @param json the json
   * @return the sq activation layer
   */
  public static SqActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new SqActivationLayer(json);
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
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
}
