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
 * The type Mean variance gradient.
 */
public class MeanVarianceGradient implements TrustRegion {
  
  private double max = Double.POSITIVE_INFINITY;
  
  @Override
  public double[] project(double[] weights, double[] point) {
    double meanWeight = ArrayUtil.mean(weights);
    double meanPoint = ArrayUtil.mean(point);
    double varWeights = ArrayUtil.mean(ArrayUtil.op(weights, x -> Math.abs(x - meanWeight)));
    double varPoint = ArrayUtil.mean(ArrayUtil.op(point, x -> Math.abs(x - meanPoint)));
    return ArrayUtil.op(weights, v -> {
      return (v - meanWeight) * (varPoint / varWeights) + meanPoint;
    });
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
  public MeanVarianceGradient setMax(double max) {
    this.max = max;
    return this;
  }
}
