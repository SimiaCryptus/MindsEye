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

package com.simiacryptus.mindseye.lang;

/**
 * Represents an evaluation record used during an optimization search.
 * We track both a record of the network's state,
 * and a record of the gradient evaluated at that point.
 */
public final class PointSample {
  /**
   * The Delta.
   */
  public final DeltaSet delta;
  /**
   * The Weights.
   */
  public final StateSet weights;
  /**
   * The Sum.
   */
  public final double sum;
  /**
   * The Count.
   */
  public final int count;
  /**
   * The Rate.
   */
  public double rate;

  /**
   * Instantiates a new Point sample.
   *
   * @param delta   the delta
   * @param weights the weights
   * @param sum     the sum
   * @param count   the count
   */
  public PointSample(DoubleBufferSet<Delta> delta, DoubleBufferSet<State> weights, double sum, int count) {
    this(delta, weights, sum, 0.0, count);
  }

  /**
   * Instantiates a new Point sample.
   *
   * @param delta   the delta
   * @param weights the weights
   * @param sum     the sum
   * @param rate    the rate
   */
  public PointSample(DoubleBufferSet<Delta> delta, DoubleBufferSet<State> weights, double sum, double rate) {
    this(delta, weights, sum, rate, 1);
  }

  /**
   * Instantiates a new Point sample.
   *
   * @param delta   the delta
   * @param weights the weights
   * @param sum     the sum
   */
  public PointSample(DoubleBufferSet<Delta> delta, DoubleBufferSet<State> weights, double sum) {
    this(delta, weights, sum, 1);
  }

  /**
   * Instantiates a new Point sample.
   *  @param delta   the delta
   * @param weights the weights
   * @param sum     the sum
   * @param rate    the rate
   * @param count   the count
   */
  public PointSample(DoubleBufferSet<Delta> delta, DoubleBufferSet<State> weights, double sum, double rate, int count) {
    assert (delta.getMap().size() == weights.getMap().size());
    this.delta = new DeltaSet(delta);
    this.weights = new StateSet(weights);
    this.sum = sum;
    this.count = count;
    this.setRate(rate);
  }

  /**
   * Add point sample.
   *
   * @param left  the left
   * @param right the right
   * @return the point sample
   */
  public static PointSample add(PointSample left, PointSample right) {
    assert (left.delta.getMap().size() == left.weights.getMap().size());
    assert (right.delta.getMap().size() == right.weights.getMap().size());
    return new PointSample(left.delta.add(right.delta),
      StateSet.union(left.weights, right.weights),
      left.sum + right.sum,
      left.count + right.count);
  }

  /**
   * Gets mean.
   *
   * @return the mean
   */
  public double getMean() {
    return sum / count;
  }
  
  @Override
  public String toString() {
    final StringBuffer sb = new StringBuffer("PointSample{");
    sb.append("avg=").append(getMean());
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
    return new PointSample(delta.copy(), weights, sum, rate, count);
  }

  /**
   * Copy full point sample.
   *
   * @return the point sample
   */
  public PointSample copyFull() {
    return new PointSample(delta.copy(), weights.copy(), sum, rate, count);
  }

  /**
   * Reset point sample.
   *
   * @return the point sample
   */
  public PointSample restore() {
    weights.stream().forEach(d -> d.restore());
    return this;
  }

  /**
   * Add point sample.
   *
   * @param right the right
   * @return the point sample
   */
  public PointSample add(PointSample right) {
    return add(this, right);
  }
}