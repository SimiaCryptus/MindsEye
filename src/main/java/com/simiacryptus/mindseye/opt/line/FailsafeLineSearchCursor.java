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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

import javax.annotation.Nonnull;

/**
 * A wrapper for a line search cursor which tracks the best-known point.
 */
public class FailsafeLineSearchCursor extends LineSearchCursorBase {
  private final LineSearchCursor direction;
  private final TrainingMonitor monitor;
  private PointSample best;
  
  /**
   * Instantiates a new Failsafe line search cursor.
   *
   * @param direction     the direction
   * @param previousPoint the previous point
   * @param monitor       the monitor
   */
  public FailsafeLineSearchCursor(final LineSearchCursor direction, @Nonnull final PointSample previousPoint, final TrainingMonitor monitor) {
    this.direction = direction;
    this.direction.addRef();
    best = previousPoint.copyFull();
    this.monitor = monitor;
  }
  
  /**
   * Accumulate.
   *
   * @param step the runStep
   */
  public synchronized void accumulate(@Nonnull final PointSample step) {
    if (null == best || best.getMean() > step.getMean()) {
      @Nonnull PointSample newValue = step.copyFull();
      if (null != this.best) {
        monitor.log(String.format("New Minimum: %s > %s", best.getMean(), step.getMean()));
        this.best.freeRef();
      }
      this.best = newValue;
    }
  }
  
  
  /**
   * Gets best.
   *
   * @param monitor the monitor
   * @return the best
   */
  public PointSample getBest(final TrainingMonitor monitor) {
    if (null != this.best) {
      best.weights.restore();
      best.addRef();
    }
    return best;
  }
  
  @Override
  public CharSequence getDirectionType() {
    return direction.getDirectionType();
  }
  
  @Override
  public DeltaSet<Layer> position(final double alpha) {
    return direction.position(alpha);
  }
  
  @Override
  public void reset() {
    direction.reset();
  }
  
  @Override
  public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
    final LineSearchPoint step = direction.step(alpha, monitor);
    accumulate(step.point);
    return step;
  }
  
  @Override
  public void _free() {
    if (null != this.best) {
      this.best.freeRef();
      this.best = null;
    }
    this.direction.freeRef();
  }
  
}
