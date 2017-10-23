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

package com.simiacryptus.mindseye.eval;

import com.simiacryptus.mindseye.lang.DeltaSet;

/**
 * The interface Trainable.
 */
public interface Trainable {
  /**
   * Measure trainable . point sample.
   *
   * @return the trainable . point sample
   */
  Trainable.PointSample measure();
  
  /**
   * Reset to full.
   */
  default void resetToFull() {
  }
  
  /**
   * Reset sampling boolean.
   *
   * @return the boolean
   */
  default boolean resetSampling() {
    return false;
  }
  
  /**
   * The type Point sample.
   */
  public final class PointSample {
    /**
     * The Delta.
     */
    public final DeltaSet delta;
    /**
     * The Weights.
     */
    public final DeltaSet weights;
    /**
     * The Value.
     */
    public final double value;
    /**
     * The Rate.
     */
    public double rate;
    
    /**
     * Instantiates a new Point sample.
     *
     * @param delta   the delta
     * @param weights the weights
     * @param value   the value
     */
    public PointSample(DeltaSet delta, DeltaSet weights, double value) {
      this(delta, weights, value, 0.0);
    }
    
    /**
     * Instantiates a new Point sample.
     *
     * @param delta   the delta
     * @param weights the weights
     * @param value   the value
     * @param rate    the rate
     */
    public PointSample(DeltaSet delta, DeltaSet weights, double value, double rate) {
      this.delta = delta;
      this.weights = weights;
      this.value = value;
      this.setRate(rate);
    }
    
    @Override
    public String toString() {
      final StringBuffer sb = new StringBuffer("PointSample{");
      sb.append("value=").append(value);
      sb.append('}');
      return sb.toString();
    }
    
    /**
     * Gets rate.
     *
     * @return the rate
     */
    public double getRate() {
      return rate;
    }
    
    /**
     * Sets rate.
     *
     * @param rate the rate
     * @return the rate
     */
    public PointSample setRate(double rate) {
      this.rate = rate;
      return this;
    }
    
    
    /**
     * Copy delta point sample.
     *
     * @return the point sample
     */
    public PointSample copyDelta() {
      return new PointSample(delta.copy(), weights, value, rate);
    }
    
    /**
     * Copy full point sample.
     *
     * @return the point sample
     */
    public PointSample copyFull() {
      return new PointSample(delta.copy(), weights.copy(), value, rate);
    }
    
    /**
     * Reset.
     */
    public void reset() {
      weights.vector().stream().forEach(d -> d.overwrite());
    }
    
    /**
     * Add point sample.
     *
     * @param right the right
     * @return the point sample
     */
    public PointSample add(PointSample right) {
      return new PointSample(this.delta.add(right.delta), this.weights, this.value + right.value);
    }
  }
}
