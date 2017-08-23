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

import com.simiacryptus.mindseye.layers.Delta;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.ArrayUtil;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.IntStream;
import static com.simiacryptus.util.ArrayUtil.*;

/**
 * The type Trust region strategy.
 */
public abstract class TrustRegionStrategy implements OrientationStrategy {
  
  
  /**
   * The Inner.
   */
  public final OrientationStrategy inner;
  private int maxHistory = 10;
  
  /**
   * Instantiates a new Trust region strategy.
   */
  public TrustRegionStrategy() {
    this(new LBFGS());
  }
  
  /**
   * Instantiates a new Trust region strategy.
   *
   * @param inner the inner
   */
  protected TrustRegionStrategy(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  private final List<PointSample> history = new LinkedList<>();
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample origin, TrainingMonitor monitor) {
    history.add(0,origin);
    while(history.size() > maxHistory) history.remove(history.size()-1);
    LineSearchCursor cursor = inner.orient(subject, origin, monitor);
    return new LineSearchCursor() {
      @Override
      public String getDirectionType() {
        return cursor.getDirectionType() + "+Trust";
      }
  
      @Override
      public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        DeltaSet currentDirection = position(alpha).write();
        PointSample measurement = measure(alpha, monitor);
        return new LineSearchPoint(measurement, dot(currentDirection.vector(), measurement.delta.vector()));
      }
  
      public PointSample measure(double alpha, TrainingMonitor monitor) {
        return cursor.measure(alpha, monitor);
      }

      @Override
      public DeltaSet position(double alpha) {
        // Restore to orginal position
        cursor.reset();
        DeltaSet innerVector = cursor.position(alpha).add(origin.weights.scale(-1));
        // Adjust new point and associated tangent
        DeltaSet currentDirection = innerVector.copy();
        innerVector.map.forEach((layer, buffer) -> {
          if (null == buffer.getDelta()) return;
          Delta deltaBuffer = currentDirection.get(layer, buffer.target);
          double[] delta = deltaBuffer.getDelta();
          double[] projected = add(deltaBuffer.target, delta);
          TrustRegion region = getRegionPolicy(layer);
          if(null != region) {
            double[][] historyData = history.stream().map((PointSample x) -> x.weights.map.get(layer).getDelta()).toArray(i -> new double[i][]);
            double[] adjusted = region.project(historyData, projected);
            if(adjusted != projected) {
              double[] correction = subtract(adjusted, projected);
              double correctionMagSq = ArrayUtil.dot(correction,correction);
              if(0 != correctionMagSq) {
                double dot = ArrayUtil.dot(delta, correction);
                double a = dot / correctionMagSq;
                if(a != -1) {
                  double[] tangent = add(delta, multiply(correction, -a));
                  assert(ArrayUtil.dot(tangent, tangent) <= ArrayUtil.dot(delta, delta));
                  for (int i = 0; i < tangent.length; i++) {
                    projected[i] = adjusted[i];
                    delta[i] = tangent[i];
                  }
                }
              }
            }
          }
          for (int i = 0; i < buffer.getDelta().length; i++) {
            buffer.target[i] = projected[i];
          }
        });
        return currentDirection;
      }
  
      @Override
      public void reset() {
        cursor.reset();
      }
    };
  }
  
  /**
   * Gets region policy.
   *
   * @param layer the layer
   * @return the region policy
   */
  public abstract TrustRegion getRegionPolicy(NNLayer layer);
  
  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(List<Delta> a, List<Delta> b) {
    assert (a.size() == b.size());
    return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
  }
  
  /**
   * Gets max history.
   *
   * @return the max history
   */
  public int getMaxHistory() {
    return maxHistory;
  }
  
  /**
   * Sets max history.
   *
   * @param maxHistory the max history
   * @return the max history
   */
  public TrustRegionStrategy setMaxHistory(int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
  }
}
