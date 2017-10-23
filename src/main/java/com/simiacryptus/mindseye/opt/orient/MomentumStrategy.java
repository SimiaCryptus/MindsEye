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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.util.ArrayUtil;

/**
 * The type Momentum strategy.
 */
public class MomentumStrategy implements OrientationStrategy {
  
  /**
   * The Inner.
   */
  public final OrientationStrategy inner;
  private double carryOver = 0.1;
  
  /**
   * Instantiates a new Momentum strategy.
   *
   * @param inner the inner
   */
  public MomentumStrategy(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  /**
   * The Prev delta.
   */
  DeltaSet prevDelta = new DeltaSet();
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor orient = inner.orient(subject, measurement, monitor);
    DeltaSet direction = ((SimpleLineSearchCursor) orient).direction;
    DeltaSet newDelta = new DeltaSet();
    direction.map.forEach((layer, delta) -> {
      Delta prevBuffer = prevDelta.get(layer, delta.target);
      newDelta.get(layer, delta.target).accumulate(ArrayUtil.add(ArrayUtil.multiply(prevBuffer.getDelta(), carryOver), delta.getDelta()));
    });
    prevDelta = newDelta;
    return new SimpleLineSearchCursor(subject, measurement, newDelta);
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  /**
   * Gets carry over.
   *
   * @return the carry over
   */
  public double getCarryOver() {
    return carryOver;
  }
  
  /**
   * Sets carry over.
   *
   * @param carryOver the carry over
   * @return the carry over
   */
  public MomentumStrategy setCarryOver(double carryOver) {
    this.carryOver = carryOver;
    return this;
  }
}
