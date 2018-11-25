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
 * The Absolute Value.
 */
@SuppressWarnings("serial")
public final class AbsActivationLayer extends SimpleActivationLayer<AbsActivationLayer> {

  /**
   * Instantiates a new Abs activation key.
   */
  public AbsActivationLayer() {
  }

  /**
   * Instantiates a new Abs activation key.
   *
   * @param id the id
   */
  protected AbsActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json abs activation key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the abs activation key
   */
  public static AbsActivationLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AbsActivationLayer(json);
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

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

}
