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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.DescribeOrientationWrapper;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * This exact line search method uses a linear interpolation of the derivative to find the extrema, where dx/dy = 0.
 * Bracketing conditions are established with logic that largely ignores derivatives, due to heuristic observations.
 */
public class QuadraticSearch implements LineSearchStrategy {
  
  private final double initialDerivFactor = 0.95;
  private double absoluteTolerance = 1e-12;
  private double currentRate = 0.0;
  private double minRate = 1e-10;
  private double relativeTolerance = 1e-2;
  private double stepSize = 1.0;
  
  /**
   * Step point sample.
   *
   * @param cursor  the cursor
   * @param monitor the monitor
   * @return the point sample
   */
  public PointSample _step(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    double thisX = 0;
    LineSearchPoint thisPoint = cursor.step(thisX, monitor);
    final LineSearchPoint initialPoint = thisPoint;
    initialPoint.addRef();
    double leftX = thisX;
    LineSearchPoint leftPoint = thisPoint;
    leftPoint.addRef();
    monitor.log(String.format("F(%s) = %s", leftX, leftPoint));
    if (0 == leftPoint.derivative) {
      initialPoint.freeRef();
      thisPoint.freeRef();
      PointSample point = leftPoint.point;
      point.addRef();
      leftPoint.freeRef();
      return point;
    }
  
    @javax.annotation.Nonnull final LocateInitialRightPoint locateInitialRightPoint = new LocateInitialRightPoint(cursor, monitor, leftPoint).apply();
    @Nonnull LineSearchPoint rightPoint = locateInitialRightPoint.getRightPoint();
    rightPoint.addRef();
    double rightX = locateInitialRightPoint.getRightX();
  
    try {
      int loops = 0;
      while (true) {
        final double a = (rightPoint.derivative - leftPoint.derivative) / (rightX - leftX);
        final double b = rightPoint.derivative - a * rightX;
        thisX = -b / a;
        final boolean isBracketed = Math.signum(leftPoint.derivative) != Math.signum(rightPoint.derivative);
        if (!Double.isFinite(thisX) || isBracketed && (leftX > thisX || rightX < thisX)) {
          thisX = (rightX + leftX) / 2;
        }
        if (!isBracketed && thisX < 0) {
          thisX = rightX * 2;
        }
        if (isSame(leftX, thisX, 1.0)) {
          monitor.log(String.format("Converged to left"));
          return filter(cursor, leftPoint.point, monitor);
        }
        else if (isSame(thisX, rightX, 1.0)) {
          monitor.log(String.format("Converged to right"));
          return filter(cursor, rightPoint.point, monitor);
        }
        thisPoint.freeRef();
        thisPoint = cursor.step(thisX, monitor);
        if (isSame(cursor, monitor, leftPoint, thisPoint)) {
          monitor.log(String.format("%s ~= %s", leftX, thisX));
          return filter(cursor, leftPoint.point, monitor);
        }
        if (isSame(cursor, monitor, thisPoint, rightPoint)) {
          monitor.log(String.format("%s ~= %s", thisX, rightX));
          return filter(cursor, rightPoint.point, monitor);
        }
        thisPoint.freeRef();
        thisPoint = cursor.step(thisX, monitor);
        boolean isLeft;
        if (!isBracketed) {
          isLeft = Math.abs(rightPoint.point.rate - thisPoint.point.rate) > Math.abs(leftPoint.point.rate - thisPoint.point.rate);
        }
        else {
          isLeft = thisPoint.derivative < 0;
        }
        //monitor._log(String.format("isLeft=%s; isBracketed=%s; leftPoint=%s; rightPoint=%s", isLeft, isBracketed, leftPoint, rightPoint));
        monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
        if (loops++ > 10) {
          monitor.log(String.format("Loops = %s", loops));
          PointSample filter = filter(cursor, thisPoint.point, monitor);
          return filter;
        }
        if (isSame(cursor, monitor, leftPoint, rightPoint)) {
          monitor.log(String.format("%s ~= %s", leftX, rightX));
          PointSample filter = filter(cursor, thisPoint.point, monitor);
          return filter;
        }
        if (isLeft) {
          if (thisPoint.point.getMean() > leftPoint.point.getMean()) {
            monitor.log(String.format("%s > %s", thisPoint.point.getMean(), leftPoint.point.getMean()));
            return filter(cursor, leftPoint.point, monitor);
          }
          if (!isBracketed && leftPoint.point.getMean() < rightPoint.point.getMean()) {
            rightX = leftX;
            if (null != rightPoint) rightPoint.freeRef();
            rightPoint = leftPoint;
            rightPoint.addRef();
          }
          if (null != leftPoint) leftPoint.freeRef();
          leftPoint = thisPoint;
          leftPoint.addRef();
          leftX = thisX;
          monitor.log(String.format("Left bracket at %s", thisX));
        }
        else {
          if (thisPoint.point.getMean() > rightPoint.point.getMean()) {
            monitor.log(String.format("%s > %s", thisPoint.point.getMean(), rightPoint.point.getMean()));
            return filter(cursor, rightPoint.point, monitor);
          }
          if (!isBracketed && rightPoint.point.getMean() < leftPoint.point.getMean()) {
            leftX = rightX;
            if (null != leftPoint) leftPoint.freeRef();
            leftPoint = rightPoint;
            leftPoint.addRef();
          }
          rightX = thisX;
          if (null != rightPoint) rightPoint.freeRef();
          rightPoint = thisPoint;
          rightPoint.addRef();
          monitor.log(String.format("Right bracket at %s", thisX));
        }
      }
    } finally {
      if (null != leftPoint) leftPoint.freeRef();
      if (null != rightPoint) rightPoint.freeRef();
      if (null != thisPoint) thisPoint.freeRef();
      if (null != initialPoint) initialPoint.freeRef();
      if (null != locateInitialRightPoint) locateInitialRightPoint.freeRef();
    }
  }
  
  private String diagnose(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final TrainingMonitor monitor, @javax.annotation.Nonnull final LineSearchPoint a, @javax.annotation.Nonnull final LineSearchPoint b) {
    final LineSearchPoint verifyA = cursor.step(a.point.rate, monitor);
    final boolean validA = isSame(a.point.getMean(), verifyA.point.getMean(), 1.0);
    monitor.log(String.format("Verify %s: %s (%s)", a.point.rate, verifyA.point.getMean(), validA));
    if (!validA) {
      DescribeOrientationWrapper.render(a.point.weights, a.point.delta);
      return "Non-Reproducable Point Found: " + a.point.rate;
    }
    final LineSearchPoint verifyB = cursor.step(b.point.rate, monitor);
    final boolean validB = isSame(b.point.getMean(), verifyB.point.getMean(), 1.0);
    monitor.log(String.format("Verify %s: %s (%s)", b.point.rate, verifyB.point.getMean(), validB));
    verifyB.freeRef();
    if (!validA && !validB) return "Non-Reproducable Function Found";
    if (validA && validB) return "Function Discontinuity Found";
    if (!validA) {
      return "Non-Reproducable Point Found: " + a.point.rate;
    }
    if (!validB) {
      return "Non-Reproducable Point Found: " + b.point.rate;
    }
    return "";
  }
  
  private PointSample filter(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final PointSample point, final TrainingMonitor monitor) {
    if (stepSize == 1.0) {
      point.addRef();
      return point;
    }
    else {
      LineSearchPoint step = cursor.step(point.rate * stepSize, monitor);
      PointSample point1 = step.point;
      point1.addRef();
      step.freeRef();
      return point1;
    }
  }
  
  /**
   * Gets absolute tolerance.
   *
   * @return the absolute tolerance
   */
  public double getAbsoluteTolerance() {
    return absoluteTolerance;
  }
  
  /**
   * Sets absolute tolerance.
   *
   * @param absoluteTolerance the absolute tolerance
   * @return the absolute tolerance
   */
  @javax.annotation.Nonnull
  public QuadraticSearch setAbsoluteTolerance(final double absoluteTolerance) {
    this.absoluteTolerance = absoluteTolerance;
    return this;
  }
  
  /**
   * Gets current rate.
   *
   * @return the current rate
   */
  public double getCurrentRate() {
    return currentRate;
  }
  
  /**
   * Sets current rate.
   *
   * @param currentRate the current rate
   * @return the current rate
   */
  @javax.annotation.Nonnull
  public QuadraticSearch setCurrentRate(final double currentRate) {
    this.currentRate = currentRate;
    return this;
  }
  
  /**
   * Gets min rate.
   *
   * @return the min rate
   */
  public double getMinRate() {
    return minRate;
  }
  
  /**
   * Sets min rate.
   *
   * @param minRate the min rate
   */
  public void setMinRate(final double minRate) {
    this.minRate = minRate;
  }
  
  /**
   * Gets relative tolerance.
   *
   * @return the relative tolerance
   */
  public double getRelativeTolerance() {
    return relativeTolerance;
  }
  
  /**
   * Sets relative tolerance.
   *
   * @param relativeTolerance the relative tolerance
   * @return the relative tolerance
   */
  @javax.annotation.Nonnull
  public QuadraticSearch setRelativeTolerance(final double relativeTolerance) {
    this.relativeTolerance = relativeTolerance;
    return this;
  }
  
  /**
   * Gets runStep size.
   *
   * @return the runStep size
   */
  public double getStepSize() {
    return stepSize;
  }
  
  /**
   * Sets runStep size.
   *
   * @param stepSize the runStep size
   * @return the runStep size
   */
  @javax.annotation.Nonnull
  public QuadraticSearch setStepSize(final double stepSize) {
    this.stepSize = stepSize;
    return this;
  }
  
  /**
   * Is same boolean.
   *
   * @param a     the a
   * @param b     the b
   * @param slack the slack
   * @return the boolean
   */
  protected boolean isSame(final double a, final double b, final double slack) {
    final double diff = Math.abs(a - b) / slack;
    final double scale = Math.max(Math.abs(a), Math.abs(b));
    return diff < absoluteTolerance || diff < scale * relativeTolerance;
  }
  
  /**
   * Is same boolean.
   *
   * @param cursor  the cursor
   * @param monitor the monitor
   * @param a       the a
   * @param b       the b
   * @return the boolean
   */
  protected boolean isSame(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final TrainingMonitor monitor, @javax.annotation.Nonnull final LineSearchPoint a, @javax.annotation.Nonnull final LineSearchPoint b) {
    if (isSame(a.point.rate, b.point.rate, 1.0)) {
      if (!isSame(a.point.getMean(), b.point.getMean(), 10.0)) {
        @javax.annotation.Nonnull final String diagnose = diagnose(cursor, monitor, a, b);
        monitor.log(diagnose);
        throw new IterativeStopException(diagnose);
      }
      return true;
    }
    else {
      return false;
    }
  }
  
  @Override
  public PointSample step(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final TrainingMonitor monitor) {
    if (currentRate < getMinRate()) {
      currentRate = getMinRate();
    }
    final PointSample pointSample = _step(cursor, monitor);
    setCurrentRate(pointSample.rate);
    return pointSample;
  }
  
  private class LocateInitialRightPoint extends ReferenceCountingBase {
    @javax.annotation.Nonnull
    private final LineSearchCursor cursor;
    @javax.annotation.Nonnull
    private final LineSearchPoint initialPoint;
    @javax.annotation.Nonnull
    private final TrainingMonitor monitor;
    private LineSearchPoint thisPoint;
    private double thisX;
  
    /**
     * Instantiates a new Locate initial right point.
     *
     * @param cursor    the cursor
     * @param monitor   the monitor
     * @param leftPoint the left point
     */
    public LocateInitialRightPoint(@javax.annotation.Nonnull final LineSearchCursor cursor, @javax.annotation.Nonnull final TrainingMonitor monitor, @javax.annotation.Nonnull final LineSearchPoint leftPoint) {
      this.cursor = cursor;
      this.monitor = monitor;
      initialPoint = leftPoint;
      thisX = getCurrentRate() > 0 ? getCurrentRate() : Math.abs(leftPoint.point.getMean() * 1e-4 / leftPoint.derivative);
      thisPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
      this.cursor.addRef();
      this.initialPoint.addRef();
    }
  
    /**
     * Apply locate initial right point.
     *
     * @return the locate initial right point
     */
    @javax.annotation.Nonnull
    public LocateInitialRightPoint apply() {
      @Nullable LineSearchPoint lastPoint = null;
      try {
        int loops = 0;
        while (true) {
          if (null != lastPoint) lastPoint.freeRef();
          lastPoint = thisPoint;
          lastPoint.addRef();
          if (isSame(cursor, monitor, initialPoint, thisPoint)) {
            monitor.log(String.format("%s ~= %s", initialPoint.point.rate, thisX));
            return this;
          }
          else if (thisPoint.point.getMean() > initialPoint.point.getMean()) {
            thisX = thisX / 13;
          }
          else if (thisPoint.derivative < initialDerivFactor * thisPoint.derivative) {
            thisX = thisX * 7;
          }
          else {
            monitor.log(String.format("%s <= %s", thisPoint.point.getMean(), initialPoint.point.getMean()));
            return this;
          }
  
          if (null != thisPoint) thisPoint.freeRef();
          thisPoint = cursor.step(thisX, monitor);
          if (isSame(cursor, monitor, lastPoint, thisPoint)) {
            monitor.log(String.format("%s ~= %s", lastPoint.point.rate, thisX));
            return this;
          }
          monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
          if (loops++ > 50) {
            monitor.log(String.format("Loops = %s", loops));
            return this;
          }
        }
      } finally {
        if (null != lastPoint) lastPoint.freeRef();
      }
    }
  
    /**
     * Gets right point.
     *
     * @return the right point
     */
    public LineSearchPoint getRightPoint() {
      return thisPoint;
    }
  
    /**
     * Gets right x.
     *
     * @return the right x
     */
    public double getRightX() {
      return thisX;
    }
    
    @Override
    protected void _free() {
      this.thisPoint.freeRef();
      this.cursor.freeRef();
      this.initialPoint.freeRef();
    }
  }
}
