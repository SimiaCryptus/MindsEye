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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.lang.PointSample;

/**
 * Data describing the result of a given training iteration evaluation.
 */
public class Step {
  /**
   * The Iteration.
   */
  public final long iteration;
  /**
   * The Point.
   */
  public final PointSample point;
  /**
   * The Time.
   */
  public final long time = System.currentTimeMillis();
  
  /**
   * Instantiates a new Step.
   *
   * @param point     the point
   * @param iteration the iteration
   */
  Step(final PointSample point, final long iteration) {
    this.point = point;
    this.iteration = iteration;
  }
}
