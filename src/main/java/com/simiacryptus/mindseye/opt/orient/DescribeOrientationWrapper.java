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

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This wrapper adds extra logging to the orientation step.
 */
public class DescribeOrientationWrapper extends OrientationStrategyBase<LineSearchCursor> {
  
  private final OrientationStrategy<? extends LineSearchCursor> inner;
  
  /**
   * Instantiates a new Describe orientation wrapper.
   *
   * @param inner the heapCopy
   */
  public DescribeOrientationWrapper(final OrientationStrategy<? extends LineSearchCursor> inner) {
    this.inner = inner;
  }
  
  /**
   * Gets id.
   *
   * @param x the x
   * @return the id
   */
  @javax.annotation.Nonnull
  public static String getId(@javax.annotation.Nonnull final DoubleBuffer<NNLayer> x) {
    final String name = x.layer.getName();
    @javax.annotation.Nonnull final String className = x.layer.getClass().getSimpleName();
    return name.contains(className) ? className : name;
//    if(x.layer instanceof PlaceholderLayer) {
//      return "Input";
//    }
//    return x.layer.toString();
  }
  
  /**
   * Render string.
   *
   * @param weightDelta the weight delta
   * @param dirDelta    the dir delta
   * @return the string
   */
  public static String render(@javax.annotation.Nonnull final DoubleBuffer<NNLayer> weightDelta, @javax.annotation.Nonnull final DoubleBuffer<NNLayer> dirDelta) {
    @javax.annotation.Nonnull final String weightString = Arrays.toString(weightDelta.getDelta());
    @javax.annotation.Nonnull final String deltaString = Arrays.toString(dirDelta.getDelta());
    return String.format("pos: %s\nvec: %s", weightString, deltaString);
  }
  
  /**
   * Render string.
   *
   * @param weights   the weights
   * @param direction the direction
   * @return the string
   */
  public static String render(@javax.annotation.Nonnull final StateSet<NNLayer> weights, @javax.annotation.Nonnull final DeltaSet<NNLayer> direction) {
    final Map<String, String> data = weights.stream()
      .collect(Collectors.groupingBy(x -> DescribeOrientationWrapper.getId(x), Collectors.toList())).entrySet().stream()
      .collect(Collectors.toMap(x -> x.getKey(), (@javax.annotation.Nonnull final Map.Entry<String, List<State<NNLayer>>> list) -> {
        final List<State<NNLayer>> deltaList = list.getValue();
        if (1 == deltaList.size()) {
          final State<NNLayer> weightDelta = deltaList.get(0);
          return DescribeOrientationWrapper.render(weightDelta, direction.getMap().get(weightDelta.layer));
        }
        else {
          return deltaList.stream().map(weightDelta -> {
            return DescribeOrientationWrapper.render(weightDelta, direction.getMap().get(weightDelta.layer));
          }).limit(10)
            .reduce((a, b) -> a + "\n" + b).orElse("");
        }
      }));
    return data.entrySet().stream().map(e -> String.format("%s = %s", e.getKey(), e.getValue()))
      .map(str -> str.replaceAll("\n", "\n\t"))
      .reduce((a, b) -> a + "\n" + b).orElse("");
  }
  
  @Override
  public LineSearchCursor orient(final Trainable subject, final PointSample measurement, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    final LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if (cursor instanceof SimpleLineSearchCursor) {
      final DeltaSet<NNLayer> direction = ((SimpleLineSearchCursor) cursor).direction;
      @Nonnull final StateSet<NNLayer> weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      final String asString = DescribeOrientationWrapper.render(weights, direction);
      monitor.log(String.format("Orientation Details: %s", asString));
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
  
  
  @Override
  protected void _free() {
    this.inner.freeRef();
  }
}
