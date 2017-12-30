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

/**
 * This constrains a weight vector based on a single hyperplane which prevents immediate increases to the L1 magnitude.
 * (Note: This region can allow effective L1 increases, if at least one weight changes sign; this allows for our entire
 * search space to be reachable.)
 */
public class LinearSumConstraint implements TrustRegion {
  private boolean permitDecrease = true;
  
  /**
   * Is permit decrease boolean.
   *
   * @return the boolean
   */
  public boolean isPermitDecrease() {
    return permitDecrease;
  }
  
  /**
   * Sets permit decrease.
   *
   * @param permitDecrease the permit decrease
   * @return the permit decrease
   */
  public LinearSumConstraint setPermitDecrease(final boolean permitDecrease) {
    this.permitDecrease = permitDecrease;
    return this;
  }
  
  @Override
  public double[] project(final double[] weights, final double[] point) {
    double deltaSum = 0;
    for (int i = 0; i < point.length; i++) {
      deltaSum += (point[i] - weights[i]) * sign(point[i]);
    }
    if (deltaSum <= 0 && permitDecrease) return point;
    deltaSum /= point.length;
    final double[] returnValue = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      returnValue[i] = point[i] - deltaSum * sign(point[i]);
    }
    return returnValue;
  }
  
  /**
   * Sign int.
   *
   * @param weight the weight
   * @return the int
   */
  public int sign(final double weight) {
    return weight > 0 ? 1 : -1;
  }
}
