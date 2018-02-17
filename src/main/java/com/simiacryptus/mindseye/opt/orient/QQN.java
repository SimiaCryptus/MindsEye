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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursorBase;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;

import javax.annotation.Nonnull;

/**
 * Quadratic Quasi-Newton optimization This method hybridizes pure gradient descent with higher-order quasinewton
 * implementations such as L-BFGS. During each iteration, a quadratic curve is interpolated which aligns with the
 * gradient's direction prediction and intersects with the quasinewton's optimal point prediction. A simple parameteric
 * quadratic function blends both heapCopy cursors into a simple nonlinear path which should combine the stability of
 * both methods.
 */
public class QQN extends OrientationStrategyBase<LineSearchCursor> {
  
  /**
   * The constant CURSOR_NAME.
   */
  public static final String CURSOR_NAME = "QQN";
  private final LBFGS inner = new LBFGS();
  
  /**
   * Gets max history.
   *
   * @return the max history
   */
  public int getMaxHistory() {
    return inner.getMaxHistory();
  }
  
  /**
   * Sets max history.
   *
   * @param maxHistory the max history
   * @return the max history
   */
  @javax.annotation.Nonnull
  public QQN setMaxHistory(final int maxHistory) {
    inner.setMaxHistory(maxHistory);
    return this;
  }
  
  /**
   * Gets min history.
   *
   * @return the min history
   */
  public int getMinHistory() {
    return inner.getMinHistory();
  }
  
  /**
   * Sets min history.
   *
   * @param minHistory the min history
   * @return the min history
   */
  @javax.annotation.Nonnull
  public QQN setMinHistory(final int minHistory) {
    inner.setMinHistory(minHistory);
    return this;
  }
  
  @Override
  public LineSearchCursor orient(@javax.annotation.Nonnull final Trainable subject, @javax.annotation.Nonnull final PointSample origin, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    inner.addToHistory(origin, monitor);
    final SimpleLineSearchCursor lbfgsCursor = inner.orient(subject, origin, monitor);
    final DeltaSet<Layer> lbfgs = lbfgsCursor.direction;
    @Nonnull final DeltaSet<Layer> gd = origin.delta.scale(-1.0);
    final double lbfgsMag = lbfgs.getMagnitude();
    final double gdMag = gd.getMagnitude();
    if (Math.abs(lbfgsMag - gdMag) / (lbfgsMag + gdMag) > 1e-2) {
      @Nonnull final DeltaSet<Layer> scaledGradient = gd.scale(lbfgsMag / gdMag);
      monitor.log(String.format("Returning Quadratic Cursor %s GD, %s QN", gdMag, lbfgsMag));
      gd.freeRef();
      return new LineSearchCursorBase() {
  
        @javax.annotation.Nonnull
        @Override
        public String getDirectionType() {
          return CURSOR_NAME;
        }
  
        @Override
        public DeltaSet<Layer> position(final double t) {
          if (!Double.isFinite(t)) throw new IllegalArgumentException();
          return scaledGradient.scale(t - t * t).add(lbfgs.scale(t * t));
        }
  
        @Override
        public void reset() {
          lbfgsCursor.reset();
        }
  
        @javax.annotation.Nonnull
        @Override
        public LineSearchPoint step(final double t, @Nonnull final TrainingMonitor monitor) {
          if (!Double.isFinite(t)) throw new IllegalArgumentException();
          reset();
          position(t).accumulate(1);
          @Nonnull final PointSample sample = subject.measure(monitor).setRate(t);
          //monitor.log(String.format("delta buffers %d %d %d %d %d", sample.delta.run.size(), origin.delta.run.size(), lbfgs.run.size(), gd.run.size(), scaledGradient.run.size()));
          inner.addToHistory(sample, monitor);
          @Nonnull final DeltaSet<Layer> tangent = scaledGradient.scale(1 - 2 * t).add(lbfgs.scale(2 * t));
          return new LineSearchPoint(sample, tangent.dot(sample.delta));
        }
  
        @Override
        public void _free() {
          scaledGradient.freeRef();
          lbfgsCursor.freeRef();
        }
      };
    }
    else {
      gd.freeRef();
      return lbfgsCursor;
    }
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
