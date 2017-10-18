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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.trainable.Trainable;

/**
 * The type Failsafe line search cursor.
 */
public class FailsafeLineSearchCursor implements LineSearchCursor {
  private final LineSearchCursor direction;
  private LineSearchPoint best;
  
  /**
   * Instantiates a new Failsafe line search cursor.
   *
   * @param direction the direction
   * @param previousPoint
   */
  public FailsafeLineSearchCursor(LineSearchCursor direction, Trainable.PointSample previousPoint) {
    this.direction = direction;
    this.best = null;
    assert 0 == previousPoint.rate;
    accumulate(new LineSearchPoint(previousPoint, Double.NaN));
  }
  
  @Override
  public String getDirectionType() {
    return direction.getDirectionType();
  }
  
  
  @Override
  public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
    LineSearchPoint step = direction.step(alpha, monitor);
    accumulate(step);
    return step;
  }
  
  public void accumulate(LineSearchPoint step) {
    if (null == this.best || this.best.point.value > step.point.value) {
      this.best = step;
    }
  }
  
  @Override
  public DeltaSet position(double alpha) {
    return direction.position(alpha);
  }
  
  @Override
  public void reset() {
    direction.reset();
  }
  
  /**
   * Gets best.
   *
   * @return the best
   */
  public LineSearchPoint getBest(TrainingMonitor monitor) {
    return null==best?null:step(best.point.rate, monitor);
  }
  
}
