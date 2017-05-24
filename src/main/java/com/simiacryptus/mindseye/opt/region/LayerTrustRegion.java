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
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.util.ArrayUtil;

import java.util.List;
import java.util.stream.IntStream;
import static com.simiacryptus.util.ArrayUtil.*;

public abstract class LayerTrustRegion implements OrientationStrategy {
  
  
  public final OrientationStrategy inner;
  
  public LayerTrustRegion() {
    this(new LBFGS());
  }
  
  protected LayerTrustRegion(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
    DeltaSet direction = ((SimpleLineSearchCursor) inner.orient(subject, measurement, monitor)).direction;
    return new LineSearchCursor() {
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
            double[] adjusted = region.project(deltaBuffer.target, projected);
            if(adjusted != projected) {
              double[] correction = subtract(adjusted, projected);
              double correctionMagSq = ArrayUtil.dot(correction,correction);
              if(0 != correctionMagSq) {
                double dot = ArrayUtil.dot(delta, correction);
                double a = dot / correctionMagSq;
                if(a != -1) {
                  double[] tangent = add(delta, multiply(correction, -a));
                  assert(ArrayUtil.dot(tangent, tangent) < ArrayUtil.dot(delta, delta));
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
        Trainable.PointSample measurement = subject.measure();
        return new LineSearchPoint(measurement, dot(currentDirection.vector(), measurement.delta.vector()));
      }
    };
  }
  
  public abstract TrustRegion getRegionPolicy(NNLayer layer);
  
  public static double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
    assert (a.size() == b.size());
    return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
  }
  
}
