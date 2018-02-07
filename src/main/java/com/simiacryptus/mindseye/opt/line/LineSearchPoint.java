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

import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;

/**
 * A particular point in a NNLayer line search phase. Contains both the high-dimensional position and derivative, and
 * the simplified one-dimensional positiion and derivative.
 */
public class LineSearchPoint extends ReferenceCountingBase {
  /**
   * The Derivative.
   */
  public final double derivative;
  /**
   * The Point.
   */
  public final PointSample point;
  
  /**
   * Instantiates a new Line search point.
   *
   * @param point      the point
   * @param derivative the derivative
   */
  public LineSearchPoint(final PointSample point, final double derivative) {
    this.point = point;
    this.point.addRef();
    this.derivative = derivative;
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    @javax.annotation.Nonnull final StringBuffer sb = new StringBuffer("LineSearchPoint{");
    sb.append("point=").append(point);
    sb.append(", derivative=").append(derivative);
    sb.append('}');
    return sb.toString();
  }
  
  @Override
  protected void _free() {
    point.freeRef();
  }
}
