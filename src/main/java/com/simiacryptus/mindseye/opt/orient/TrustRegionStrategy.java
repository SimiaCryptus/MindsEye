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
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursorBase;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.util.ArrayUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A generalization of the OWL-QN algorithm, this wrapping strategy projects an heapCopy cursor to the interior of a
 * trust region, which can be defined per-key. Any simple orientation strategy can be used as the heapCopy, most
 * commonly either GD or LBFGS. Many trust regions can be defined; see the com.simiacryptus.mindseye.opt.region
 * package.
 */
public abstract class TrustRegionStrategy extends OrientationStrategyBase<LineSearchCursor> {


  /**
   * The Inner.
   */
  public final OrientationStrategy<? extends SimpleLineSearchCursor> inner;
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
   * @param inner the heapCopy
   */
  protected TrustRegionStrategy(final OrientationStrategy<? extends SimpleLineSearchCursor> inner) {
    this.inner = inner;
  }

  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(@Nonnull final List<DoubleBuffer<UUID>> a, @Nonnull final List<DoubleBuffer<UUID>> b) {
    assert a.size() == b.size();
    return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
  }

  @Override
  protected void _free() {
    this.inner.freeRef();
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
  @Nonnull
  public TrustRegionStrategy setMaxHistory(final int maxHistory) {
    this.maxHistory = maxHistory;
    return this;
  }

  /**
   * Gets the Trust Region for a particular LayerBase
   *
   * @param layer the key
   * @return the region policy
   */
  public abstract TrustRegion getRegionPolicy(Layer layer);

  @Nonnull
  @Override
  public LineSearchCursor orient(@Nonnull final Trainable subject, final PointSample origin, final TrainingMonitor monitor) {
    history.add(0, origin);
    while (history.size() > maxHistory) {
      history.remove(history.size() - 1);
    }
    final SimpleLineSearchCursor cursor = inner.orient(subject, origin, monitor);
    return new LineSearchCursorBase() {
      @Nonnull
      @Override
      public CharSequence getDirectionType() {
        return cursor.getDirectionType() + "+Trust";
      }

      @Nonnull
      @Override
      public DeltaSet<UUID> position(final double alpha) {
        reset();
        @Nonnull final DeltaSet<UUID> adjustedPosVector = cursor.position(alpha);
        project(adjustedPosVector, new TrainingMonitor());
        return adjustedPosVector;
      }

      public Layer toLayer(UUID id) {
        return ((DAGNetwork)subject.getLayer()).getLayersById().get(id);
      }

      @Nonnull
      public DeltaSet<UUID> project(@Nonnull final DeltaSet<UUID> deltaIn, final TrainingMonitor monitor) {
        final DeltaSet<UUID> originalAlphaDerivative = cursor.direction;
        @Nonnull final DeltaSet<UUID> newAlphaDerivative = originalAlphaDerivative.copy();
        deltaIn.getMap().forEach((id, buffer) -> {
          @Nullable final double[] delta = buffer.getDelta();
          if (null == delta) return;
          final double[] currentPosition = buffer.target;
          @Nullable final double[] originalAlphaD = originalAlphaDerivative.get(id, currentPosition).getDeltaAndFree();
          @Nullable final double[] newAlphaD = newAlphaDerivative.get(id, currentPosition).getDeltaAndFree();
          @Nonnull final double[] proposedPosition = ArrayUtil.add(currentPosition, delta);
          final TrustRegion region = getRegionPolicy(toLayer(id));
          if (null != region) {
            final Stream<double[]> zz = history.stream().map((@Nonnull final PointSample x) -> {
              final DoubleBuffer<UUID> d = x.weights.getMap().get(id);
              @Nullable final double[] z = null == d ? null : d.getDelta();
              return z;
            });
            final double[] projectedPosition = region.project(zz.filter(x -> null != x).toArray(i -> new double[i][]), proposedPosition);
            if (projectedPosition != proposedPosition) {
              for (int i = 0; i < projectedPosition.length; i++) {
                delta[i] = projectedPosition[i] - currentPosition[i];
              }
              @Nonnull final double[] normal = ArrayUtil.subtract(projectedPosition, proposedPosition);
              final double normalMagSq = ArrayUtil.dot(normal, normal);
//              monitor.log(String.format("%s: evalInputDelta = %s, projectedPosition = %s, proposedPosition = %s, currentPosition = %s, normalMagSq = %s", key,
//                ArrayUtil.dot(evalInputDelta,evalInputDelta),
//                ArrayUtil.dot(projectedPosition,projectedPosition),
//                ArrayUtil.dot(proposedPosition,proposedPosition),
//                ArrayUtil.dot(currentPosition,currentPosition),
//                normalMagSq));
              if (0 < normalMagSq) {
                final double a = ArrayUtil.dot(originalAlphaD, normal);
                if (a != -1) {
                  @Nonnull final double[] tangent = ArrayUtil.add(originalAlphaD, ArrayUtil.multiply(normal, -a / normalMagSq));
                  for (int i = 0; i < tangent.length; i++) {
                    newAlphaD[i] = tangent[i];
                  }
//                  double newAlphaDerivSq = ArrayUtil.dot(tangent, tangent);
//                  double originalAlphaDerivSq = ArrayUtil.dot(originalAlphaD, originalAlphaD);
//                  assert(newAlphaDerivSq <= originalAlphaDerivSq);
//                  assert(Math.abs(ArrayUtil.dot(tangent, normal)) <= 1e-4);
//                  monitor.log(String.format("%s: normalMagSq = %s, newAlphaDerivSq = %s, originalAlphaDerivSq = %s", key, normalMagSq, newAlphaDerivSq, originalAlphaDerivSq));
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

      @Nonnull
      @Override
      public LineSearchPoint step(final double alpha, final TrainingMonitor monitor) {
        cursor.reset();
        @Nonnull final DeltaSet<UUID> adjustedPosVector = cursor.position(alpha);
        @Nonnull final DeltaSet<UUID> adjustedGradient = project(adjustedPosVector, monitor);
        adjustedPosVector.accumulate(1);
        adjustedPosVector.freeRef();
        @Nonnull final PointSample sample = subject.measure(monitor).setRate(alpha);
        double dot = adjustedGradient.dot(sample.delta);
        adjustedGradient.freeRef();
        LineSearchPoint lineSearchPoint = new LineSearchPoint(sample, dot);
        sample.freeRef();
        return lineSearchPoint;
      }

      @Override
      public void _free() {
        cursor.freeRef();
      }
    };
  }

  @Override
  public void reset() {
    inner.reset();
  }
}
