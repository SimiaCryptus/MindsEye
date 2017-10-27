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

/**
 * The type Sigmoid activation layer.
 */
public final class GaussianActivationLayer extends SimpleActivationLayer<GaussianActivationLayer> {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("mean", mean);
    json.addProperty("stddev", stddev);
    return json;
  }

  /**
   * From json sigmoid activation layer.
   *
   * @param json the json
   * @return the sigmoid activation layer
   */
  public static GaussianActivationLayer fromJson(JsonObject json) {
    return new GaussianActivationLayer(json);
  }

  /**
   * Instantiates a new Sigmoid activation layer.
   *
   * @param id the id
   */
  protected GaussianActivationLayer(JsonObject id) {
    super(id);
    mean = id.get("mean").getAsDouble();
    stddev = id.get("stddev").getAsDouble();
  }
  
  private static final double MIN_X = -20;
  private static final double MAX_X = -MIN_X;
  private static final double MAX_F = Math.exp(MAX_X);
  private static final double MIN_F = Math.exp(MIN_X);
  /**
   *
   */
  private static final long serialVersionUID = -1676818127036480927L;
  
  private double mean;
  private double stddev;
  
  /**
   * Instantiates a new Sigmoid activation layer.
   */
  public GaussianActivationLayer(double mean, double stddev) {
    this.mean = mean;
    this.stddev = stddev;
  }
  
  @Override
  protected final void eval(final double x, final double[] results) {
    final double minDeriv = 0;
    final double c = x - mean;
    final double s2 = stddev * stddev;
    final double s3 = stddev * s2;
    final double k = Math.sqrt(2 * Math.PI);
    final double e = exp(-((c * c) / (2 * s2)));
    double d = e * c / (s3 * k);
    double f = e / (stddev * k);
    // double d = f * (1 - f);
    if (!Double.isFinite(d) || d < minDeriv) {
      d = minDeriv;
    }
    assert Double.isFinite(d);
    assert minDeriv <= Math.abs(d);
    results[0] = f;
    results[1] = -d;
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
  
}
