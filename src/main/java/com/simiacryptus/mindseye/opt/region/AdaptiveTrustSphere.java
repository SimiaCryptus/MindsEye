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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;

/**
 * This trust region uses recent position history to define an ellipsoid volume for the n+1 line search
 */
public class AdaptiveTrustSphere implements TrustRegion {

  private int divisor = 5;
  private int lookback = 10;

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
  @Nonnull
  public AdaptiveTrustSphere setDivisor(final int divisor) {
    this.divisor = divisor;
    return this;
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
  @Nonnull
  public AdaptiveTrustSphere setLookback(final int lookback) {
    this.lookback = lookback;
    return this;
  }

  /**
   * Length double.
   *
   * @param weights the weights
   * @return the double
   */
  public double length(@Nonnull final double[] weights) {
    return ArrayUtil.magnitude(weights);
  }

  @Nonnull
  @Override
  public double[] project(@Nonnull final double[][] history, @Nonnull final double[] point) {
    final double[] weights = history[0];
    @Nonnull final double[] delta = ArrayUtil.subtract(point, weights);
    final double distance = ArrayUtil.magnitude(delta);
    if (history.length < lookback + 1) return point;
    final double max = ArrayUtil.magnitude(ArrayUtil.subtract(weights, history[lookback])) / divisor;
    return distance > max ? ArrayUtil.add(weights, ArrayUtil.multiply(delta, max / distance)) : point;
  }
}
