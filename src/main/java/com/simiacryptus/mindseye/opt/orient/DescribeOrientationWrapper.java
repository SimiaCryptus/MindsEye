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
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.DoubleBuffer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.State;
import com.simiacryptus.mindseye.lang.StateSet;
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
  @Nonnull
  public static CharSequence getId(@Nonnull final DoubleBuffer<Layer> x) {
    final String name = x.layer.getName();
    @Nonnull final CharSequence className = x.layer.getClass().getSimpleName();
    return name.contains(className) ? className : name;
//    if(x.layer instanceof PlaceholderLayer) {
//      return "Input";
//    }
//    return x.layer.toStream();
  }
  
  /**
   * Render string.
   *
   * @param weightDelta the weight delta
   * @param dirDelta    the dir delta
   * @return the string
   */
  public static CharSequence render(@Nonnull final DoubleBuffer<Layer> weightDelta, @Nonnull final DoubleBuffer<Layer> dirDelta) {
    @Nonnull final CharSequence weightString = Arrays.toString(weightDelta.getDelta());
    @Nonnull final CharSequence deltaString = Arrays.toString(dirDelta.getDelta());
    return String.format("pos: %s\nvec: %s", weightString, deltaString);
  }
  
  /**
   * Render string.
   *
   * @param weights   the weights
   * @param direction the direction
   * @return the string
   */
  public static CharSequence render(@Nonnull final StateSet<Layer> weights, @Nonnull final DeltaSet<Layer> direction) {
    final Map<CharSequence, CharSequence> data = weights.stream()
      .collect(Collectors.groupingBy(x -> DescribeOrientationWrapper.getId(x), Collectors.toList())).entrySet().stream()
      .collect(Collectors.toMap(x -> x.getKey(), (@Nonnull final Map.Entry<CharSequence, List<State<Layer>>> list) -> {
        final List<State<Layer>> deltaList = list.getValue();
        if (1 == deltaList.size()) {
          final State<Layer> weightDelta = deltaList.get(0);
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
  public LineSearchCursor orient(final Trainable subject, final PointSample measurement, @Nonnull final TrainingMonitor monitor) {
    final LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if (cursor instanceof SimpleLineSearchCursor) {
      final DeltaSet<Layer> direction = ((SimpleLineSearchCursor) cursor).direction;
      @Nonnull final StateSet<Layer> weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      final CharSequence asString = DescribeOrientationWrapper.render(weights, direction);
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
