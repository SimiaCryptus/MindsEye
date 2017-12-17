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
 * This constraint ensures that the L2 magnitude of the weight delta cannot exceed a simple threshold.
 * A simpler version of AdaptiveTrustSphere, it places a limit on the step size for a given layer.
 */
public class DistanceConstraint implements TrustRegion {
  
  private double max = Double.POSITIVE_INFINITY;
  
  /**
   * Gets max.
   *
   * @return the max
   */
  public double getMax() {
    return max;
  }
  
  /**
   * Sets max.
   *
   * @param max the max
   * @return the max
   */
  public DistanceConstraint setMax(final double max) {
    this.max = max;
    return this;
  }
  
  /**
   * Length double.
   *
   * @param weights the weights
   * @return the double
   */
  public double length(final double[] weights) {
    return ArrayUtil.magnitude(weights);
  }
  
  @Override
  public double[] project(final double[] weights, final double[] point) {
    final double[] delta = ArrayUtil.subtract(point, weights);
    final double distance = ArrayUtil.magnitude(delta);
    return distance > max ? ArrayUtil.add(weights, ArrayUtil.multiply(delta, max / distance)) : point;
  }
}
