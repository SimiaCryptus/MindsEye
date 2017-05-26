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

public class GrowthSphere implements TrustRegion {
  private double growthFactor = 1.5;
  private double minRadius = 1;
  
  @Override
  public double[] project(double[] weights, double[] point) {
    double stateMagnitude = length(weights);
    double frontier = getRadius(stateMagnitude);
    double pointMag = length(point);
    if (pointMag < frontier) return point;
    return ArrayUtil.multiply(point, frontier / pointMag);
  }
  
  public double length(double[] weights) {
    return Math.sqrt(ArrayUtil.dot(weights, weights));
  }
  
  public double getRadius(double stateMagnitude) {
    return Math.max(minRadius, stateMagnitude * growthFactor);
  }
  
  public double getGrowthFactor() {
    return growthFactor;
  }
  
  public GrowthSphere setGrowthFactor(double growthFactor) {
    this.growthFactor = growthFactor;
    return this;
  }
  
  public double getMinRadius() {
    return minRadius;
  }
  
  public GrowthSphere setMinRadius(double minRadius) {
    this.minRadius = minRadius;
    return this;
  }
}
