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

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchPoint;
import com.simiacryptus.mindseye.eval.Trainable;

/**
 * The type Validating orientation strategy.
 */
public class ValidatingOrientationStrategy implements OrientationStrategy {
  
  private final OrientationStrategy inner;
  
  /**
   * Instantiates a new Validating orientation strategy.
   *
   * @param inner the inner
   */
  public ValidatingOrientationStrategy(OrientationStrategy inner) {
    this.inner = inner;
  }
  
  @Override
  public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
    LineSearchCursor cursor = inner.orient(subject, measurement, monitor);
    return new ValidatingLineSearchCursor(cursor);
  }
  
  @Override
  public void reset() {
    inner.reset();
  }
  
  private static class ValidatingLineSearchCursor implements LineSearchCursor {
    private final LineSearchCursor cursor;

    /**
     * Instantiates a new Validating line search cursor.
     *
     * @param cursor the cursor
     */
    public ValidatingLineSearchCursor(LineSearchCursor cursor) {
      this.cursor = cursor;
    }
    
    @Override
    public String getDirectionType() {
      return cursor.getDirectionType();
    }
    
    @Override
    public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
      LineSearchPoint primaryPoint = cursor.step(alpha, monitor);
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
    public void test(TrainingMonitor monitor, LineSearchPoint primaryPoint, double probeSize) {
      double tolerance = 1e-4;
      double alpha = primaryPoint.point.rate;
      double probeAlpha = alpha + ((primaryPoint.point.value * probeSize) / primaryPoint.derivative);
      if (!Double.isFinite(probeAlpha) || probeAlpha == alpha) {
        probeAlpha = alpha + probeSize;
      }
      LineSearchPoint probePoint = cursor.step(probeAlpha, monitor);
      double dy = probePoint.point.value - primaryPoint.point.value;
      double dx = probeAlpha - alpha;
      double measuredDerivative = dy / dx;
      monitor.log(String.format("%s vs (%s, %s); probe=%s", measuredDerivative, primaryPoint.derivative, probePoint.derivative, probeSize));
    }

    private int compare(double a, double b, double tol) {
      double c = 2 * (a - b) / (a + b);
      if (c < -tol) return -1;
      if (c > tol) return 1;
      return 0;
    }

    @Override
    public DeltaSet position(double alpha) {
      return cursor.position(alpha);
    }
    
    @Override
    public void reset() {
      cursor.reset();
    }
  }
}
