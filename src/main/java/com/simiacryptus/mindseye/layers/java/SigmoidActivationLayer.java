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
 * The type Sigmoid activation layer.
 */
public final class SigmoidActivationLayer extends SimpleActivationLayer<SigmoidActivationLayer> {
  
  private static final double MIN_X = -20;
  private static final double MAX_X = -MIN_X;
  private static final double MAX_F = Math.exp(MAX_X);
  private static final double MIN_F = Math.exp(MIN_X);
  private boolean balanced = true;
  
  /**
   * Instantiates a new Sigmoid activation layer.
   *
   * @param id the id
   */
  protected SigmoidActivationLayer(JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }
  
  /**
   * Instantiates a new Sigmoid activation layer.
   */
  public SigmoidActivationLayer() {
  }
  
  /**
   * From json sigmoid activation layer.
   *
   * @param json the json
   * @return the sigmoid activation layer
   */
  public static SigmoidActivationLayer fromJson(JsonObject json) {
    return new SigmoidActivationLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
    return json;
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double ex = exp(x);
    final double ex1 = 1 + ex;
    double d = ex / (ex1 * ex1);
    double f = 1 / (1 + 1. / ex);
    // double d = f * (1 - f);
    if (!Double.isFinite(d) || d < minDeriv) {
      d = minDeriv;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    if (isBalanced()) {
      d = 2 * d;
      f = 2 * f - 1;
    }
    results[0] = f;
    results[1] = d;
  }
  
  private double exp(final double x) {
    if (x < MIN_X) {
      return MIN_F;
    }
    if (x > MAX_X) {
      return MAX_F;
    }
    return Math.exp(x);
  }
  
  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return this.balanced;
  }
  
  /**
   * Sets balanced.
   *
   * @param balanced the balanced
   * @return the balanced
   */
  public SigmoidActivationLayer setBalanced(final boolean balanced) {
    this.balanced = balanced;
    return this;
  }
}
