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
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class DescribeOrientationWrapper implements OrientationStrategy {
  
  private final OrientationStrategy inner;
  
  public DescribeOrientationWrapper(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    if(cursor instanceof SimpleLineSearchCursor) {
      DeltaSet direction = ((SimpleLineSearchCursor) cursor).direction;
      DeltaSet weights = ((SimpleLineSearchCursor) cursor).origin.weights;
      String asString = render(weights, direction);
      monitor.log(String.format("Orientation Details: %s", asString));
    } else {
      monitor.log(String.format("Non-simple cursor: %s", cursor));
    }
    return cursor;
  }
  
  public static String render(DeltaSet weights, DeltaSet direction) {
    Map<String, String> data = weights.stream()
      .collect(Collectors.groupingBy(x -> getId(x), Collectors.toList())).entrySet().stream()
      .collect(Collectors.toMap(x -> x.getKey(), (Map.Entry<String, List<Delta>> list) -> {
        List<Delta> deltaList = list.getValue();
        if (1 == deltaList.size()) {
          Delta weightDelta = deltaList.get(0);
          return render(weightDelta, direction.getMap().get(weightDelta.layer));
        } else {
          return deltaList.stream().map(weightDelta -> {
            return render(weightDelta, direction.getMap().get(weightDelta.layer));
          }).limit(10)
            .reduce((a, b) -> a + "\n" + b).orElse("");
        }
      }));
    return data.entrySet().stream().map(e -> String.format("%s = %s", e.getKey(), e.getValue()))
      .map(str->str.replaceAll("\n","\n\t"))
      .reduce((a, b) -> a + "\n" + b).orElse("");
  }
  
  public static String render(Delta weightDelta, Delta dirDelta) {
    String weightString = Arrays.toString(weightDelta.getDelta());
    String deltaString = Arrays.toString(dirDelta.getDelta());
    return String.format("pos: %s\nvec: %s", weightString, deltaString);
  }
  
  public static String getId(Delta x) {
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
