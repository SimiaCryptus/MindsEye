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
 * A y=log(abs(x)) activation function. Note the discontinuity at 0. 
 */
@SuppressWarnings("serial")
public final class LogActivationLayer extends SimpleActivationLayer<LogActivationLayer> {
  
  /**
   * Instantiates a new Log activation layer.
   */
  public LogActivationLayer() {
  }
  
  /**
   * Instantiates a new Log activation layer.
   *
   * @param id the id
   */
  protected LogActivationLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json log activation layer.
   *
   * @param json the json
   * @return the log activation layer
   */
  public static LogActivationLayer fromJson(final JsonObject json) {
    return new LogActivationLayer(json);
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    if (x < 0) {
      eval(-x, results);
      results[0] *= 1;
      results[1] *= -1;
    }
    else if (x > 0) {
      final double minDeriv = 0;
      final double d = 0 == x ? Double.NaN : 1 / x;
      final double f = 0 == x ? Double.NEGATIVE_INFINITY : Math.log(Math.abs(x));
      assert Double.isFinite(d);
      assert minDeriv <= Math.abs(d);
      results[0] = f;
      results[1] = d;
    }
    else {
      results[0] = 0;
      results[1] = 0;
    }
  }
  
  @Override
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
}
