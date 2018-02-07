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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursorBase;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import org.jetbrains.annotations.NotNull;

/**
 * This strategy uses finite-difference methods to estimate a numerical derivative, and compares it with the derivative
 * supplied by the heapCopy's cursor. This is a diagnostic tool; extra processing is used to estimate derivatives which
 * should agree with the programmatic derivatives to an appropriate degree.
 */
public class ValidatingOrientationWrapper extends OrientationStrategyBase<LineSearchCursor> {
  
  @Override
  protected void _free() {
    this.inner.freeRef();
  }
  
  private final OrientationStrategy<? extends LineSearchCursor> inner;
  
  /**
   * Instantiates a new Validating orientation strategy.
   *
   * @param inner the heapCopy
   */
  public ValidatingOrientationWrapper(final OrientationStrategy<? extends LineSearchCursor> inner) {
    this.inner = inner;
  }
  
  @Override
  public @NotNull LineSearchCursor orient(final Trainable subject, final PointSample measurement, final TrainingMonitor monitor) {
    final LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    return new ValidatingLineSearchCursor(cursor);
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  private static class ValidatingLineSearchCursor extends LineSearchCursorBase {
    private final LineSearchCursor cursor;
  
    /**
     * Instantiates a new Validating line search cursor.
     *
     * @param cursor the cursor
     */
    public ValidatingLineSearchCursor(final LineSearchCursor cursor) {
      this.cursor = cursor;
      this.cursor.addRef();
    }

    @Override
    public String getDirectionType() {
      return cursor.getDirectionType();
    }

    @Override
    public DeltaSet<NNLayer> position(final double alpha) {
      return cursor.position(alpha);
    }
  
    @Override
    public void reset() {
      cursor.reset();
    }
  
    @Override
    public LineSearchPoint step(final double alpha, final @NotNull TrainingMonitor monitor) {
      final LineSearchPoint primaryPoint = cursor.step(alpha, monitor);
      //monitor.log(String.format("f(%s) = %s",alpha, primaryPoint.point.value));
      test(monitor, primaryPoint, 1e-3);
      test(monitor, primaryPoint, 1e-4);
      test(monitor, primaryPoint, 1e-6);
      return primaryPoint;
    }
  
    /**
     * Test.
     *
     * @param monitor      the monitor
     * @param primaryPoint the primary point
     * @param probeSize    the probe size
     */
    public void test(final @NotNull TrainingMonitor monitor, final @NotNull LineSearchPoint primaryPoint, final double probeSize) {
      final double alpha = primaryPoint.point.rate;
      double probeAlpha = alpha + primaryPoint.point.sum * probeSize / primaryPoint.derivative;
      if (!Double.isFinite(probeAlpha) || probeAlpha == alpha) {
        probeAlpha = alpha + probeSize;
      }
      final LineSearchPoint probePoint = cursor.step(probeAlpha, monitor);
      final double dy = probePoint.point.sum - primaryPoint.point.sum;
      final double dx = probeAlpha - alpha;
      final double measuredDerivative = dy / dx;
      monitor.log(String.format("%s vs (%s, %s); probe=%s", measuredDerivative, primaryPoint.derivative, probePoint.derivative, probeSize));
    }
    
    @Override
    protected void _free() {
      cursor.freeRef();
    }
  }
}
