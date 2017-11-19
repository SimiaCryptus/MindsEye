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
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
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
  private final List<PointSample> history = new LinkedList<>();
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
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, PointSample origin, TrainingMonitor monitor) {
    history.add(0, origin);
    while (history.size() > maxHistory) history.remove(history.size() - 1);
    SimpleLineSearchCursor cursor = (SimpleLineSearchCursor) inner.orient(subject, origin, monitor);
    return new LineSearchCursor() {
      @Override
      public String getDirectionType() {
        return cursor.getDirectionType() + "+Trust";
      }
      
      @Override
      public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        cursor.reset();
        DeltaSet adjustedPosVector = cursor.position(alpha);
        DeltaSet adjustedGradient = project(adjustedPosVector, monitor);
        adjustedPosVector.accumulate();
        PointSample sample = subject.measure(true, monitor).setRate(alpha);
        return new LineSearchPoint(sample, adjustedGradient.dot(sample.delta));
      }
      
      @Override
      public DeltaSet position(double alpha) {
        reset();
        DeltaSet adjustedPosVector = cursor.position(alpha);
        project(adjustedPosVector, new TrainingMonitor());
        return adjustedPosVector;
      }
      
      public DeltaSet project(DeltaSet deltaSet, TrainingMonitor monitor) {
        DeltaSet originalAlphaDerivative = cursor.direction;
        DeltaSet newAlphaDerivative = originalAlphaDerivative.copy();
        deltaSet.getMap().forEach((layer, buffer) -> {
          double[] delta = buffer.getDelta();
          if (null == delta) return;
          double[] currentPosition = buffer.target;
          double[] originalAlphaD = originalAlphaDerivative.get(layer, currentPosition).getDelta();
          double[] newAlphaD = newAlphaDerivative.get(layer, currentPosition).getDelta();
          double[] proposedPosition = add(currentPosition, delta);
          TrustRegion region = getRegionPolicy(layer);
          if (null != region) {
            double[][] historyData = history.stream().map((PointSample x) -> x.weights.getMap().get(layer).getDelta()).toArray(i -> new double[i][]);
            double[] projectedPosition = region.project(historyData, proposedPosition);
            if (projectedPosition != proposedPosition) {
              for (int i = 0; i < projectedPosition.length; i++) {
                delta[i] = projectedPosition[i] - currentPosition[i];
              }
              double[] normal = subtract(projectedPosition, proposedPosition);
              double normalMagSq = ArrayUtil.dot(normal, normal);
//              monitor.log(String.format("%s: delta = %s, projectedPosition = %s, proposedPosition = %s, currentPosition = %s, normalMagSq = %s", layer,
//                ArrayUtil.dot(delta,delta),
//                ArrayUtil.dot(projectedPosition,projectedPosition),
//                ArrayUtil.dot(proposedPosition,proposedPosition),
//                ArrayUtil.dot(currentPosition,currentPosition),
//                normalMagSq));
              if (0 < normalMagSq) {
                double a = ArrayUtil.dot(originalAlphaD, normal);
                if (a != -1) {
                  double[] tangent = add(originalAlphaD, multiply(normal, -a / normalMagSq));
                  for (int i = 0; i < tangent.length; i++) {
                    newAlphaD[i] = tangent[i];
                  }
//                  double newAlphaDerivSq = ArrayUtil.dot(tangent, tangent);
//                  double originalAlphaDerivSq = ArrayUtil.dot(originalAlphaD, originalAlphaD);
//                  assert(newAlphaDerivSq <= originalAlphaDerivSq);
//                  assert(Math.abs(ArrayUtil.dot(tangent, normal)) <= 1e-4);
//                  monitor.log(String.format("%s: normalMagSq = %s, newAlphaDerivSq = %s, originalAlphaDerivSq = %s", layer, normalMagSq, newAlphaDerivSq, originalAlphaDerivSq));
                }
              }
              
              
            }
          }
        });
        return newAlphaDerivative;
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
