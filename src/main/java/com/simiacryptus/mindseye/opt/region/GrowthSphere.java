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
 * This trust region restricts a weight vector so that it cannot increase in L2 magnitude beyond a certian amount each
 * iteration. This effectively generates a spherical trust region centered on the origin apply the current position X
 * distance into the sphere interior. A growth factor must be allowed for to provide convergence behavior so that the
 * search algorithm can reach all possible weightings.
 */
public class GrowthSphere implements TrustRegion {
  private boolean allowShrink = true;
  private double growthFactor = 1.5;
  private double minRadius = 1;

  /**
   * Gets growth factor.
   *
   * @return the growth factor
   */
  public double getGrowthFactor() {
    return growthFactor;
  }

  /**
   * Sets growth factor.
   *
   * @param growthFactor the growth factor
   * @return the growth factor
   */
  @Nonnull
  public GrowthSphere setGrowthFactor(final double growthFactor) {
    this.growthFactor = growthFactor;
    return this;
  }

  /**
   * Gets min radius.
   *
   * @return the min radius
   */
  public double getMinRadius() {
    return minRadius;
  }

  /**
   * Sets min radius.
   *
   * @param minRadius the min radius
   * @return the min radius
   */
  @Nonnull
  public GrowthSphere setMinRadius(final double minRadius) {
    this.minRadius = minRadius;
    return this;
  }

  /**
   * Gets radius.
   *
   * @param stateMagnitude the state magnitude
   * @return the radius
   */
  public double getRadius(final double stateMagnitude) {
    return Math.max(minRadius, stateMagnitude * growthFactor);
  }

  /**
   * Is allow shrink boolean.
   *
   * @return the boolean
   */
  public boolean isAllowShrink() {
    return allowShrink;
  }

  /**
   * Sets allow shrink.
   *
   * @param allowShrink the allow shrink
   * @return the allow shrink
   */
  @Nonnull
  public GrowthSphere setAllowShrink(final boolean allowShrink) {
    this.allowShrink = allowShrink;
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
  public double[] project(@Nonnull final double[] weights, @Nonnull final double[] point) {
    final double stateMagnitude = length(weights);
    final double frontier = getRadius(stateMagnitude);
    final double pointMag = length(point);
    if (pointMag < frontier && allowShrink) return point;
    return ArrayUtil.multiply(point, frontier / pointMag);
  }
}
