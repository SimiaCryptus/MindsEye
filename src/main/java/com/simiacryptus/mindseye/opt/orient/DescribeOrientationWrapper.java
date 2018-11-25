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
import java.util.UUID;
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
  public static CharSequence getId(@Nonnull final DoubleBuffer<UUID> x) {
    return x.key.toString();
  }

  /**
   * Render string.
   *
   * @param weightDelta the weight evalInputDelta
   * @param dirDelta    the dir evalInputDelta
   * @return the string
   */
  public static CharSequence render(@Nonnull final DoubleBuffer<UUID> weightDelta, @Nonnull final DoubleBuffer<UUID> dirDelta) {
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
  public static CharSequence render(@Nonnull final StateSet<UUID> weights, @Nonnull final DeltaSet<UUID> direction) {
    final Map<CharSequence, CharSequence> data = weights.stream()
        .collect(Collectors.groupingBy(x -> DescribeOrientationWrapper.getId(x), Collectors.toList())).entrySet().stream()
        .collect(Collectors.toMap(x -> x.getKey(), (@Nonnull final Map.Entry<CharSequence, List<State<UUID>>> list) -> {
          final List<State<UUID>> deltaList = list.getValue();
          if (1 == deltaList.size()) {
            final State<UUID> weightDelta = deltaList.get(0);
            return DescribeOrientationWrapper.render(weightDelta, direction.getMap().get(weightDelta.key));
          } else {
            return deltaList.stream().map(weightDelta -> {
              return DescribeOrientationWrapper.render(weightDelta, direction.getMap().get(weightDelta.key));
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
      final DeltaSet<UUID> direction = ((SimpleLineSearchCursor) cursor).direction;
      @Nonnull final StateSet<UUID> weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      final CharSequence asString = DescribeOrientationWrapper.render(weights, direction);
      monitor.log(String.format("Orientation Details: %s", asString));
    } else {
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
