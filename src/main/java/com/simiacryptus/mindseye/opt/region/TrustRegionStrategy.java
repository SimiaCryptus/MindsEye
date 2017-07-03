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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.ArrayUtil;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import static com.simiacryptus.util.ArrayUtil.*;

public abstract class TrustRegionStrategy implements OrientationStrategy {
  
  
  public final OrientationStrategy inner;
  private int maxHistory = 10;
  
  public TrustRegionStrategy() {
    this(new LBFGS());
  }
  
  protected TrustRegionStrategy(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  private final List<PointSample> history = new LinkedList<>();
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample measurement, TrainingMonitor monitor) {
    history.add(0,measurement);
    while(history.size() > maxHistory) history.remove(history.size()-1);
    LineSearchCursor orient = inner.orient(subject, measurement, monitor);
    DeltaSet direction = ((SimpleLineSearchCursor) orient).direction;
    return new LineSearchCursor() {
      @Override
      public String getDirectionType() {
        return orient.getDirectionType() + "+Trust";
      }
  
      @Override
      public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        // Restore to orginal position
        measurement.weights.vector().stream().forEach(d -> d.overwrite());
        // Adjust new point and associated tangent
        DeltaSet currentDirection = direction.copy();
        direction.map.forEach((layer, buffer) -> {
          if (null == buffer.delta) return;
          DeltaBuffer deltaBuffer = currentDirection.get(layer, buffer.target);
          double[] delta = multiply(deltaBuffer.delta, alpha);
          double[] projected = add(deltaBuffer.target, delta);
          TrustRegion region = getRegionPolicy(layer);
          if(null != region) {
            double[][] historyData = history.stream().map(x -> x.weights.map.get(layer).delta).toArray(i -> new double[i][]);
            double[] adjusted = region.project(historyData, projected);
            if(adjusted != projected) {
              double[] correction = subtract(adjusted, projected);
              double correctionMagSq = ArrayUtil.dot(correction,correction);
              if(0 != correctionMagSq) {
                double dot = ArrayUtil.dot(delta, correction);
                double a = dot / correctionMagSq;
                if(a != -1) {
                  double[] tangent = add(delta, multiply(correction, -a));
                  //assert(ArrayUtil.dot(tangent, tangent) <= ArrayUtil.dot(delta, delta));
                  for (int i = 0; i < tangent.length; i++) {
                    projected[i] = adjusted[i];
                    deltaBuffer.delta[i] = tangent[i];
                  }
                }
              }
            }
          }
          for (int i = 0; i < buffer.delta.length; i++) {
            buffer.target[i] = projected[i];
          }
        });
        // Execute measurement and return
        PointSample measurement = subject.measure();
        return new LineSearchPoint(measurement, dot(currentDirection.vector(), measurement.delta.vector()));
      }
    };
  }
  
  public abstract TrustRegion getRegionPolicy(NNLayer layer);
  
  public static double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
    assert (a.size() == b.size());
    return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
  }
  
  public int getMaxHistory() {
    return maxHistory;
  }
  
  public TrustRegionStrategy setMaxHistory(int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
  }
}
