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
 * Created by Andrew Charneski on 5/23/2017.
 */
public class SingleOrthant implements TrustRegion {
  @Override
  public double[] project(double[] weights, double[] point) {
    double[] returnValue = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      int positionSign = sign(weights[i]);
      int directionSign = sign(point[i]);
      returnValue[i] = (0 != positionSign && positionSign != directionSign) ? 0 : point[i];
    }
    return returnValue;
  }
  
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
    zeroTol = zeroTol;
  }
  
  /**
   * Sign int.
   *
   * @param weight the weight
   * @return the int
   */
  public int sign(double weight) {
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
  
  private final double zeroTol = 1e-20;
  
}
