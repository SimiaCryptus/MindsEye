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
 * This highly-constrained region allows ONLY changes to the
 * mean/stddev of the weight vector components.
 * Experimental; no proven use case.
 */
public class MeanVarianceGradient implements TrustRegion {
  
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
  public MeanVarianceGradient setMax(final double max) {
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
    final double meanWeight = ArrayUtil.mean(weights);
    final double meanPoint = ArrayUtil.mean(point);
    final double varWeights = ArrayUtil.mean(ArrayUtil.op(weights, x -> Math.abs(x - meanWeight)));
    final double varPoint = ArrayUtil.mean(ArrayUtil.op(point, x -> Math.abs(x - meanPoint)));
    return ArrayUtil.op(weights, v -> {
      return (v - meanWeight) * (varPoint / varWeights) + meanPoint;
    });
  }
}
