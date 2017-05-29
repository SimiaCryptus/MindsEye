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

public class DistanceConstraint implements TrustRegion {
  
  private double max = Double.POSITIVE_INFINITY;
  
  @Override
  public double[] project(double[] weights, double[] point) {
    double[] delta = ArrayUtil.subtract(point, weights);
    double distance = ArrayUtil.magnitude(delta);
    return distance>max?ArrayUtil.add(weights, ArrayUtil.multiply(delta, max / distance)):point;
  }
  
  public double length(double[] weights) {
    return ArrayUtil.magnitude(weights);
  }
  
  public double getMax() {
    return max;
  }
  
  public DistanceConstraint setMax(double max) {
    this.max = max;
    return this;
  }
}