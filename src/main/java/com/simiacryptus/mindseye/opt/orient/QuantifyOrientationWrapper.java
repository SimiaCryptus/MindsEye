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

import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.StateSet;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.DoubleBuffer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.data.DoubleStatistics;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An orientation wrapper which adds additional log statements.
 */
public class QuantifyOrientationWrapper implements OrientationStrategy<LineSearchCursor> {
  
  private final OrientationStrategy<? extends LineSearchCursor> inner;
  
  /**
   * Instantiates a new Quantify orientation wrapper.
   *
   * @param inner the inner
   */
  public QuantifyOrientationWrapper(OrientationStrategy<? extends LineSearchCursor> inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if(cursor instanceof SimpleLineSearchCursor) {
      DeltaSet direction = ((SimpleLineSearchCursor) cursor).direction;
      StateSet weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      Map<String, String> dataMap = weights.stream()
        .collect(Collectors.groupingBy(x -> getId(x), Collectors.toList())).entrySet().stream()
        .collect(Collectors.toMap(x -> x.getKey(), list -> {
          List<Double> doubleList = list.getValue().stream().map(weightDelta -> {
            DoubleBuffer dirDelta = direction.getMap().get(weightDelta.layer);
            double denominator = weightDelta.deltaStatistics().rms();
            double numerator = null == dirDelta ? 0 : dirDelta.deltaStatistics().rms();
            return (numerator / (0 == denominator ? 1 : denominator));
          }).collect(Collectors.toList());
          if (1 == doubleList.size()) return Double.toString(doubleList.get(0));
          return new DoubleStatistics().accept(doubleList.stream().mapToDouble(x -> x).toArray()).toString();
        }));
      monitor.log(String.format("Line search stats: %s", dataMap));
    } else {
      monitor.log(String.format("Non-simple cursor: %s", cursor));
    }
    return cursor;
  }
  
  /**
   * Gets id.
   *
   * @param x the x
   * @return the id
   */
  public String getId(DoubleBuffer x) {
    String name = x.layer.getName();
    String className = x.layer.getClass().getSimpleName();
    return name.contains(className)?className:name;
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
