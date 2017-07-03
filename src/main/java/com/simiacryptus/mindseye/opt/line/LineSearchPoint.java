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

import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;

/**
 * Created by Andrew Charneski on 5/9/2017.
 */
public class LineSearchPoint {
  public final PointSample point;
  public final double derivative;
  
  public LineSearchPoint(PointSample point, double derivative) {
    this.point = point;
    this.derivative = derivative;
  }
  
  @Override
  public String toString() {
    final StringBuffer sb = new StringBuffer("LineSearchPoint{");
    sb.append("point=").append(point);
    sb.append(", derivative=").append(derivative);
    sb.append('}');
    return sb.toString();
  }
}
