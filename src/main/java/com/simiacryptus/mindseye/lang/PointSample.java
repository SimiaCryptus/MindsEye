/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import org.jetbrains.annotations.NotNull;

/**
 * Represents an evaluation record used during an optimization search. We track both a record of the network's state,
 * and a record of the gradient evaluated at that point.
 */
public final class PointSample extends ReferenceCountingBase {
  /**
   * The Count.
   */
  public final int count;
  /**
   * The Delta.
   */
  public final @NotNull DeltaSet<NNLayer> delta;
  /**
   * The Sum.
   */
  public final double sum;
  /**
   * The Weights.
   */
  public final @NotNull StateSet<NNLayer> weights;
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
   * @param rate    the rate
   * @param count   the count
   */
  public PointSample(final @NotNull DeltaSet<NNLayer> delta, final @NotNull StateSet<NNLayer> weights, final double sum, final double rate, final int count) {
    assert delta.getMap().size() == weights.getMap().size();
    this.delta = new DeltaSet<>(delta);
    this.weights = new StateSet<>(weights);
    assert delta.getMap().keySet().stream().allMatch(x -> weights.getMap().containsKey(x));
    this.sum = sum;
    this.count = count;
    setRate(rate);
  }
  
  /**
   * Add point sample.
   *
   * @param left  the left
   * @param right the right
   * @return the point sample
   */
  public static PointSample add(final @NotNull PointSample left, final @NotNull PointSample right) {
    assert left.delta.getMap().size() == left.weights.getMap().size();
    assert right.delta.getMap().size() == right.weights.getMap().size();
    assert left.rate == right.rate;
    return new PointSample(left.delta.add(right.delta),
                           StateSet.union(left.weights, right.weights),
                           left.sum + right.sum,
                           left.rate,
                           left.count + right.count);
  }
  
  /**
   * Add point sample.
   *
   * @param right the right
   * @return the point sample
   */
  public PointSample add(final @NotNull PointSample right) {
    return PointSample.add(this, right);
  }
  
  /**
   * Add in place point sample.
   *
   * @param right the right
   * @return the point sample
   */
  public PointSample addInPlace(final @NotNull PointSample right) {
    assert delta.getMap().size() == weights.getMap().size();
    assert right.delta.getMap().size() == right.weights.getMap().size();
    assert rate == right.rate;
    return new PointSample(delta.addInPlace(right.delta),
                           StateSet.union(weights, right.weights),
                           sum + right.sum,
                           rate,
                           count + right.count);
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
  public @NotNull PointSample copyFull() {
    @NotNull DeltaSet<NNLayer> deltaCopy = delta.copy();
    @NotNull StateSet<NNLayer> weightsCopy = weights.copy();
    @NotNull PointSample pointSample = new PointSample(deltaCopy, weightsCopy, sum, rate, count);
    deltaCopy.freeRef();
    weightsCopy.freeRef();
    return pointSample;
  }
  
  /**
   * Gets mean.
   *
   * @return the mean
   */
  public double getMean() {
    return sum / count;
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
  public @NotNull PointSample setRate(final double rate) {
    this.rate = rate;
    return this;
  }
  
  /**
   * Normalize point sample.
   *
   * @return the point sample
   */
  public @NotNull PointSample normalize() {
    if (count == 1) {
      this.addRef();
      return this;
    }
    else {
      @NotNull DeltaSet<NNLayer> scale = delta.scale(1.0 / count);
      @NotNull PointSample pointSample = new PointSample(scale, weights, sum / count, rate, 1);
      scale.freeRef();
      return pointSample;
    }
  }
  
  /**
   * Reset point sample.
   *
   * @return the point sample
   */
  public @NotNull PointSample restore() {
    weights.stream().forEach(d -> d.restore());
    return this;
  }
  
  /**
   * Backup point sample.
   *
   * @return the point sample
   */
  public @NotNull PointSample backup() {
    weights.stream().forEach(d -> d.backup());
    return this;
  }
  
  @Override
  public String toString() {
    final @NotNull StringBuffer sb = new StringBuffer("PointSample{");
    sb.append("avg=").append(getMean());
    sb.append('}');
    return sb.toString();
  }
  
  @Override
  protected void _free() {
    this.weights.freeRef();
    this.delta.freeRef();
  }
}
