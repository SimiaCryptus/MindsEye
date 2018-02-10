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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.util.data.DoubleStatistics;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An orientation wrapper which adds additional log statements.
 */
public class QuantifyOrientationWrapper extends OrientationStrategyBase<LineSearchCursor> {
  
  private final OrientationStrategy<? extends LineSearchCursor> inner;
  
  /**
   * Instantiates a new Quantify orientation wrapper.
   *
   * @param inner the heapCopy
   */
  public QuantifyOrientationWrapper(final OrientationStrategy<? extends LineSearchCursor> inner) {
    this.inner = inner;
  }
  
  @Override
  protected void _free() {
    inner.freeRef();
  }
  
  /**
   * Gets id.
   *
   * @param x the x
   * @return the id
   */
  @javax.annotation.Nonnull
  public String getId(@javax.annotation.Nonnull final DoubleBuffer<NNLayer> x) {
    final String name = x.layer.getName();
    @javax.annotation.Nonnull final String className = x.layer.getClass().getSimpleName();
    return name.contains(className) ? className : name;
//    if(x.layer instanceof PlaceholderLayer) {
//      return "Input";
//    }
//    return x.layer.toString();
  }
  
  @Override
  public LineSearchCursor orient(final Trainable subject, final PointSample measurement, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    final LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if (cursor instanceof SimpleLineSearchCursor) {
      final DeltaSet<NNLayer> direction = ((SimpleLineSearchCursor) cursor).direction;
      @Nonnull final StateSet<NNLayer> weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      final Map<String, String> dataMap = weights.stream()
        .collect(Collectors.groupingBy(x -> getId(x), Collectors.toList())).entrySet().stream()
        .collect(Collectors.toMap(x -> x.getKey(), list -> {
          final List<Double> doubleList = list.getValue().stream().map(weightDelta -> {
            final DoubleBuffer<NNLayer> dirDelta = direction.getMap().get(weightDelta.layer);
            final double denominator = weightDelta.deltaStatistics().rms();
            final double numerator = null == dirDelta ? 0 : dirDelta.deltaStatistics().rms();
            return numerator / (0 == denominator ? 1 : denominator);
          }).collect(Collectors.toList());
          if (1 == doubleList.size())
            return Double.toString(doubleList.get(0));
          return new DoubleStatistics().accept(doubleList.stream().mapToDouble(x -> x).toArray()).toString();
        }));
      monitor.log(String.format("Line search stats: %s", dataMap));
    }
    else {
      monitor.log(String.format("Non-simple cursor: %s", cursor));
    }
    return cursor;
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  
}
