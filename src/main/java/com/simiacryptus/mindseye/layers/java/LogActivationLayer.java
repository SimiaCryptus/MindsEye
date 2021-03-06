/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * A y=log(abs(x)) activation function. Note the discontinuity at 0.
 */
@SuppressWarnings("serial")
public final class LogActivationLayer extends SimpleActivationLayer<LogActivationLayer> {

  /**
   * Instantiates a new Log activation key.
   */
  public LogActivationLayer() {
  }

  /**
   * Instantiates a new Log activation key.
   *
   * @param id the id
   */
  protected LogActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json log activation key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the log activation key
   */
  public static LogActivationLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LogActivationLayer(json);
  }

  @Override
  protected final void eval(final double x, final double[] results) {
    if (x < 0) {
      eval(-x, results);
      results[0] *= 1;
      results[1] *= -1;
    } else if (x > 0) {
      final double minDeriv = 0;
      final double d = 0 == x ? Double.NaN : 1 / x;
      final double f = 0 == x ? Double.NEGATIVE_INFINITY : Math.log(Math.abs(x));
      assert Double.isFinite(d);
      assert minDeriv <= Math.abs(d);
      results[0] = f;
      results[1] = d;
    } else {
      results[0] = 0;
      results[1] = 0;
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

}
