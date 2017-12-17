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
 * A Single-orthant trust region. These are used in
 * OWL-QN to proven effect in training sparse models
 * where an exact value of zero for many weights is desired.
 */
public class SingleOrthant implements TrustRegion {
  private double zeroTol = 1e-20;
  
  /**
   * Gets zero tol.
   *
   * @return the zero tol
   */
  public double getZeroTol() {
    return zeroTol;
  }
  
  /**
   * Sets zero tol.
   *
   * @param zeroTol the zero tol
   */
  public void setZeroTol(double zeroTol) {
    this.zeroTol = zeroTol;
  }
  
  @Override
  public double[] project(final double[] weights, final double[] point) {
    final double[] returnValue = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      final int positionSign = sign(weights[i]);
      final int directionSign = sign(point[i]);
      returnValue[i] = 0 != positionSign && positionSign != directionSign ? 0 : point[i];
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
    if (weight > zeroTol) {
      return 1;
    }
    else if (weight < -zeroTol) {
    }
    else {
      return -1;
    }
    return 0;
  }
  
}
