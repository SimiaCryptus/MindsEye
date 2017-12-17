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

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

/**
 * A basic line search cursor representing a linear parametric path.
 */
public class SimpleLineSearchCursor implements LineSearchCursor {
  /**
   * The Direction.
   */
  public final DeltaSet<NNLayer> direction;
  /**
   * The Origin.
   */
  public final PointSample origin;
  /**
   * The Subject.
   */
  public final Trainable subject;
  private String type = "";
  
  /**
   * Instantiates a new Simple line search cursor.
   *
   * @param subject   the subject
   * @param origin    the origin
   * @param direction the direction
   */
  public SimpleLineSearchCursor(final Trainable subject, final PointSample origin, final DeltaSet<NNLayer> direction) {
    this.origin = origin.copyFull();
    this.direction = direction;
    this.subject = subject;
  }
  
  @Override
  public String getDirectionType() {
    return type;
  }
  
  /**
   * Sets direction type.
   *
   * @param type the type
   * @return the direction type
   */
  public SimpleLineSearchCursor setDirectionType(final String type) {
    this.type = type;
    return this;
  }
  
  @Override
  public DeltaSet<NNLayer> position(final double alpha) {
    return direction.scale(alpha);
  }
  
  @Override
  public void reset() {
    origin.restore();
  }
  
  @Override
  public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
    if (!Double.isFinite(alpha)) throw new IllegalArgumentException();
    reset();
    if (0.0 != alpha) {
      direction.accumulate(alpha);
    }
    final PointSample sample = subject.measure(true, monitor).setRate(alpha);
    final double dot = direction.dot(sample.delta);
    return new LineSearchPoint(sample, dot);
  }
}
