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
 * The type Growth sphere.
 */
public class GrowthSphere implements TrustRegion {
  private double growthFactor = 1.5;
  private double minRadius = 1;
  private boolean allowShrink = true;
  
  @Override
  public double[] project(double[] weights, double[] point) {
    double stateMagnitude = length(weights);
    double frontier = getRadius(stateMagnitude);
    double pointMag = length(point);
    if (pointMag < frontier && allowShrink) return point;
    return ArrayUtil.multiply(point, frontier / pointMag);
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
   * Gets radius.
   *
   * @param stateMagnitude the state magnitude
   * @return the radius
   */
  public double getRadius(double stateMagnitude) {
    return Math.max(minRadius, stateMagnitude * growthFactor);
  }
  
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
  public GrowthSphere setGrowthFactor(double growthFactor) {
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
  public GrowthSphere setMinRadius(double minRadius) {
    this.minRadius = minRadius;
    return this;
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
  public GrowthSphere setAllowShrink(boolean allowShrink) {
    this.allowShrink = allowShrink;
    return this;
  }
}
