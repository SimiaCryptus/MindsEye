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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.util.ArrayUtil;

/**
 * The type Adaptive trust sphere.
 */
public class AdaptiveTrustSphere implements TrustRegion {
  
  private int lookback = 10;
  private int divisor = 5;
  
  @Override
  public double[] project(double[][] history, double[] point) {
    double[] weights = history[0];
    double[] delta = ArrayUtil.subtract(point, weights);
    double distance = ArrayUtil.magnitude(delta);
    if (history.length < lookback + 1) return point;
    double max = ArrayUtil.magnitude(ArrayUtil.subtract(weights, history[lookback])) / divisor;
    return distance > max ? ArrayUtil.add(weights, ArrayUtil.multiply(delta, max / distance)) : point;
  }
  
  /**
   * Length double.
   *
   * @param weights the weights
   * @return the double
   */
  public double length(double[] weights) {
    return ArrayUtil.magnitude(weights);
  }
  
  /**
   * Gets lookback.
   *
   * @return the lookback
   */
  public int getLookback() {
    return lookback;
  }
  
  /**
   * Sets lookback.
   *
   * @param lookback the lookback
   * @return the lookback
   */
  public AdaptiveTrustSphere setLookback(int lookback) {
    this.lookback = lookback;
    return this;
  }
  
  /**
   * Gets divisor.
   *
   * @return the divisor
   */
  public int getDivisor() {
    return divisor;
  }
  
  /**
   * Sets divisor.
   *
   * @param divisor the divisor
   * @return the divisor
   */
  public AdaptiveTrustSphere setDivisor(int divisor) {
    this.divisor = divisor;
    return this;
  }
}
