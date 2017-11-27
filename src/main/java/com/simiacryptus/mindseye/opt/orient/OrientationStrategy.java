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

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;

/**
 * Base class for the "orientation" optimization sub-component.
 * This class interprets the result of the differential function evaluation result at the start of a training iteration,
 * transforming the multi-dimensional point-vector-function entity into a one-dimensional search sub-problem.
 */
public interface OrientationStrategy<T extends LineSearchCursor> {
  
  /**
   * Orient line search cursor.
   *
   * @param subject     the subject
   * @param measurement the measurement
   * @param monitor     the monitor
   * @return the line search cursor
   */
  T orient(Trainable subject, PointSample measurement, TrainingMonitor monitor);
  
  /**
   * Reset.
   */
  void reset();
}
