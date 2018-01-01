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

import java.util.Map;

/**
 * The classic activation function, either "sigmoid" or "tanh" dependiong on the setting of "balanced". If
 * balanced==true, the function exhibits odd symmetry (f(x) == -f(-x)) If balanced==false, the function is bounded to
 * (0,1)
 */
@SuppressWarnings("serial")
public final class SigmoidActivationLayer extends SimpleActivationLayer<SigmoidActivationLayer> {
  
  private static final double MIN_X = -20;
  private static final double MAX_X = -SigmoidActivationLayer.MIN_X;
  private static final double MAX_F = Math.exp(SigmoidActivationLayer.MAX_X);
  private static final double MIN_F = Math.exp(SigmoidActivationLayer.MIN_X);
  private boolean balanced = true;
  
  /**
   * Instantiates a new Sigmoid activation layer.
   */
  public SigmoidActivationLayer() {
  }
  
  /**
   * Instantiates a new Sigmoid activation layer.
   *
   * @param id the id
   */
  protected SigmoidActivationLayer(final JsonObject id) {
    super(id);
    balanced = id.get("balanced").getAsBoolean();
  }
  
  /**
   * From json sigmoid activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sigmoid activation layer
   */
  public static SigmoidActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new SigmoidActivationLayer(json);
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
    if (x < SigmoidActivationLayer.MIN_X) {
      return SigmoidActivationLayer.MIN_F;
    }
    if (x > SigmoidActivationLayer.MAX_X) {
      return SigmoidActivationLayer.MAX_F;
    }
    return Math.exp(x);
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("balanced", balanced);
    return json;
  }
  
  /**
   * Is balanced boolean.
   *
   * @return the boolean
   */
  public boolean isBalanced() {
    return balanced;
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
