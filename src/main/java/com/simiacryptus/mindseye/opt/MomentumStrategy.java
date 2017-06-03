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

import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.ArrayUtil;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.IntStream;

import static com.simiacryptus.util.ArrayUtil.*;

public class MomentumStrategy implements OrientationStrategy {
  
  public final OrientationStrategy inner;
  private double carryOver = 0.1;
  
  public MomentumStrategy(OrientationStrategy inner) {
    this.inner = inner;
  }

  DeltaSet prevDelta = new DeltaSet();
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor orient = inner.orient(subject, measurement, monitor);
    DeltaSet direction = ((SimpleLineSearchCursor) orient).direction;
    DeltaSet newDelta = new DeltaSet();
    direction.map.forEach((layer, delta)->{
      DeltaBuffer prevBuffer = prevDelta.get(layer, delta.target);
      newDelta.get(layer, delta.target).accumulate(ArrayUtil.add(ArrayUtil.multiply(prevBuffer.delta, carryOver), delta.delta));
    });
    prevDelta = newDelta;
    return new SimpleLineSearchCursor(subject, measurement, newDelta);
  }
  
  public double getCarryOver() {
    return carryOver;
  }
  
  public MomentumStrategy setCarryOver(double carryOver) {
    this.carryOver = carryOver;
    return this;
  }
}
