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
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.data.DoubleStatistics;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class QuantifyOrientationWrapper implements OrientationStrategy {
  
  private final OrientationStrategy inner;
  
  public QuantifyOrientationWrapper(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if(cursor instanceof SimpleLineSearchCursor) {
      DeltaSet direction = ((SimpleLineSearchCursor) cursor).direction;
      DeltaSet weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      monitor.log(String.format("Line search stats: %s", weights.stream()
        .collect(Collectors.groupingBy(x -> getId(x), Collectors.toList())).entrySet().stream()
        .collect(Collectors.toMap(x -> x.getKey(), list -> {
          List<Double> doubleList = list.getValue().stream().map(weightDelta -> {
            Delta dirDelta = direction.getMap().get(weightDelta.layer);
            return dirDelta.rms() / weightDelta.rms();
          }).collect(Collectors.toList());
          if(1 == doubleList.size()) return Double.toString(doubleList.get(0));
          return new DoubleStatistics().accept(doubleList.stream().mapToDouble(x->x).toArray()).toString();
      }))));
    } else {
      monitor.log(String.format("Non-simple cursor: %s", cursor));
    }
    return cursor;
  }
  
  public String getId(Delta x) {
    return x.layer.getClass().getSimpleName();
//    if(x.layer instanceof PlaceholderLayer) {
//      return "Input";
//    }
//    return x.layer.toString();
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  
}
