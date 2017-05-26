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

public class LinearSumConstraint implements TrustRegion {
  @Override
  public double[] project(double[] weights, double[] point) {
    double deltaSum = 0;
    for (int i = 0; i < point.length; i++) {
      deltaSum += (point[i] - weights[i]) * sign(point[i]);
    }
    if(deltaSum <= 0) return point;
    deltaSum /= point.length;
    double[] returnValue = new double[point.length];
    for (int i = 0; i < point.length; i++) {
      returnValue[i] = point[i] - deltaSum * sign(point[i]);
    }
    return returnValue;
  }
  
  public int sign(double weight) {
    return (weight > 0)?1:-1;
  }
  
}