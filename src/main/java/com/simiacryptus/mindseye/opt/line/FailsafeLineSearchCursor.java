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

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

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
   */
  public FailsafeLineSearchCursor(LineSearchCursor direction) {
    this.direction = direction;
    this.best = null;
  }
  
  @Override
  public String getDirectionType() {
    return direction.getDirectionType();
  }
  
  
  @Override
  public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
    LineSearchPoint step = direction.step(alpha, monitor);
    if (null == getBest() || getBest().point.value > step.point.value) {
      this.best = step;
    }
    return step;
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
  public LineSearchPoint getBest() {
    return best;
  }
  
}
