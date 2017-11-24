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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.DescribeOrientationWrapper;

/**
 * The type Quadratic search.
 */
public class QuadraticSearch implements LineSearchStrategy {
  
  private final double initialDerivFactor = 0.95;
  private final int maxIterations = 100;
  private double absoluteTolerance = 1e-12;
  private double relativeTolerance = 1e-2;
  private double currentRate = 0.0;
  private double minRate = 1e-10;
  private double stepSize = 1.0;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    if (currentRate < getMinRate()) currentRate = getMinRate();
    PointSample pointSample = _step(cursor, monitor);
    setCurrentRate(pointSample.rate);
    return pointSample;
  }
  
  /**
   * Step point sample.
   *
   * @param cursor  the cursor
   * @param monitor the monitor
   * @return the point sample
   */
  public PointSample _step(LineSearchCursor cursor, TrainingMonitor monitor) {
    double thisX = 0;
    LineSearchPoint thisPoint = cursor.step(thisX, monitor);
    LineSearchPoint initialPoint = thisPoint;
    double leftX = thisX;
    LineSearchPoint leftPoint = thisPoint;
    monitor.log(String.format("F(%s) = %s", leftX, leftPoint));
    if (0 == leftPoint.derivative) return leftPoint.point;
    
    LocateInitialRightPoint locateInitialRightPoint = new LocateInitialRightPoint(cursor, monitor, leftPoint).apply();
    LineSearchPoint rightPoint = locateInitialRightPoint.getRightPoint();
    double rightX = locateInitialRightPoint.getRightX();
    
    int loops = 0;
    while (true) {
      double a = (rightPoint.derivative - leftPoint.derivative) / (rightX - leftX);
      double b = rightPoint.derivative - a * rightX;
      thisX = -b / a;
      boolean isBracketed = Math.signum(leftPoint.derivative) != Math.signum(rightPoint.derivative);
      if (!Double.isFinite(thisX) || isBracketed && (leftX > thisX || rightX < thisX)) {
        thisX = (rightX + leftX) / 2;
      }
      if (!isBracketed && thisX < 0) {
        thisX = rightX * 2;
      }
      if (isSame(leftX, thisX, 1.0)) {
        monitor.log(String.format("Converged to left"));
        return filter(cursor, leftPoint.point, monitor);
      } else if (isSame(thisX, rightX, 1.0)) {
        monitor.log(String.format("Converged to right"));
        return filter(cursor, rightPoint.point, monitor);
      }
      thisPoint = cursor.step(thisX, monitor);
      if (isSame(cursor, monitor, leftPoint, thisPoint)) {
        monitor.log(String.format("%s ~= %s", leftX, thisX));
        return filter(cursor, leftPoint.point, monitor);
      }
      if (isSame(cursor, monitor, thisPoint, rightPoint)) {
        monitor.log(String.format("%s ~= %s", thisX, rightX));
        return filter(cursor, rightPoint.point, monitor);
      }
      thisPoint = cursor.step(thisX, monitor);
      boolean isLeft;
      if (!isBracketed) {
        isLeft = Math.abs(rightPoint.point.rate - thisPoint.point.rate) > Math.abs(leftPoint.point.rate - thisPoint.point.rate);
      }
      else {
        isLeft = thisPoint.derivative < 0;
      }
      monitor.log(String.format("isLeft=%s; isBracketed=%s; leftPoint=%s; rightPoint=%s", isLeft, isBracketed, leftPoint, rightPoint));
      monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
      if (loops++ > 100) {
        monitor.log(String.format("Loops = %s", loops));
        return filter(cursor, thisPoint.point, monitor);
      }
      if (isSame(cursor, monitor, leftPoint, rightPoint)) {
        monitor.log(String.format("%s ~= %s", leftX, rightX));
        return filter(cursor, thisPoint.point, monitor);
      }
      if (isLeft) {
        if (thisPoint.point.getMean() > leftPoint.point.getMean()) {
          monitor.log(String.format("%s > %s", thisPoint.point.getMean(), leftPoint.point.getMean()));
          return filter(cursor, leftPoint.point, monitor);
        }
        if (!isBracketed && leftPoint.point.getMean() < rightPoint.point.getMean()) {
          rightX = leftX;
          rightPoint = leftPoint;
        }
        leftPoint = thisPoint;
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
          leftPoint = rightPoint;
        }
        rightX = thisX;
        rightPoint = thisPoint;
        monitor.log(String.format("Right bracket at %s", thisX));
      }
    }
  }
  
  private PointSample filter(LineSearchCursor cursor, PointSample point, TrainingMonitor monitor) {
    if (stepSize == 1.0) return point;
    return cursor.step(point.rate * stepSize, monitor).point;
  }
  
  /**
   * Is same boolean.
   *
   * @param a the a
   * @param b the b
   * @param slack
   * @return the boolean
   */
  protected boolean isSame(double a, double b, double slack) {
    double diff = Math.abs(a - b) / slack;
    double scale = Math.max(Math.abs(a), Math.abs(b));
    return diff < absoluteTolerance || diff < (scale * relativeTolerance);
  }
  
  protected boolean isSame(LineSearchCursor cursor, TrainingMonitor monitor, LineSearchPoint a, LineSearchPoint b) {
    if(isSame(a.point.rate, b.point.rate, 1.0)) {
      if(!isSame(a.point.getMean(), b.point.getMean(), 10.0)) {
        String diagnose = diagnose(cursor, monitor, a, b);
        monitor.log(diagnose);
        throw new IterativeStopException(diagnose);
      }
      return true;
    } else {
      return false;
    }
  }
  
  private String diagnose(LineSearchCursor cursor, TrainingMonitor monitor, LineSearchPoint a, LineSearchPoint b) {
    LineSearchPoint verifyA = cursor.step(a.point.rate, monitor);
    boolean validA = isSame(a.point.getMean(), verifyA.point.getMean(), 1.0);
    monitor.log(String.format("Verify %s: %s (%s)", a.point.rate, verifyA.point.getMean(), validA));
    if(!validA) {
      DescribeOrientationWrapper.render(a.point.weights,a.point.delta);
      return ("Non-Reproducable Point Found: " + a.point.rate);
    }
    LineSearchPoint verifyB = cursor.step(b.point.rate, monitor);
    boolean validB = isSame(b.point.getMean(), verifyB.point.getMean(), 1.0);
    monitor.log(String.format("Verify %s: %s (%s)", b.point.rate, verifyB.point.getMean(), validB));
    if(!validA && !validB) return ("Non-Reproducable Function Found");
    if(validA && validB)  return ("Function Discontinuity Found");
    if(!validA) {
      return ("Non-Reproducable Point Found: " + a.point.rate);
    }
    if(!validB) {
      return ("Non-Reproducable Point Found: " + b.point.rate);
    }
    return "";
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
  public QuadraticSearch setAbsoluteTolerance(double absoluteTolerance) {
    this.absoluteTolerance = absoluteTolerance;
    return this;
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
  public QuadraticSearch setRelativeTolerance(double relativeTolerance) {
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
  public QuadraticSearch setStepSize(double stepSize) {
    this.stepSize = stepSize;
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
  public QuadraticSearch setCurrentRate(double currentRate) {
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
  public void setMinRate(double minRate) {
    this.minRate = minRate;
  }
  
  private class LocateInitialRightPoint {
    private final LineSearchCursor cursor;
    private final TrainingMonitor monitor;
    private final LineSearchPoint initialPoint;
    private double thisX;
    private LineSearchPoint thisPoint;
  
    /**
     * Instantiates a new Locate initial right point.
     *
     * @param cursor    the cursor
     * @param monitor   the monitor
     * @param leftPoint the left point
     */
    public LocateInitialRightPoint(LineSearchCursor cursor, TrainingMonitor monitor, LineSearchPoint leftPoint) {
      this.cursor = cursor;
      this.monitor = monitor;
      this.initialPoint = leftPoint;
      thisX = getCurrentRate() > 0 ? getCurrentRate() : Math.abs(leftPoint.point.getMean() * 1e-4 / leftPoint.derivative);
      thisPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
    }
  
    /**
     * Gets right x.
     *
     * @return the right x
     */
    public double getRightX() {
      return thisX;
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
     * Apply locate initial right point.
     *
     * @return the locate initial right point
     */
    public LocateInitialRightPoint apply() {
      LineSearchPoint lastPoint = thisPoint;
      int loops = 0;
      while (true) {
        if (isSame(cursor, monitor, initialPoint, thisPoint)) {
          monitor.log(String.format("%s ~= %s", initialPoint.point.rate, thisX));
          return this;
        } else if (thisPoint.point.getMean() > initialPoint.point.getMean()) {
          thisX = thisX / 100;
        } else if (thisPoint.derivative < initialDerivFactor * thisPoint.derivative) {
          thisX = thisX * 7;
        } else {
          monitor.log(String.format("%s <= %s", thisPoint.point.getMean(), initialPoint.point.getMean()));
          return this;
        }
        thisPoint = cursor.step(thisX, monitor);
        if (isSame(cursor, monitor, lastPoint, thisPoint)) {
          monitor.log(String.format("%s ~= %s", lastPoint.point.rate, thisX));
          return this;
        }
        lastPoint = thisPoint;
        monitor.log(String.format("F(%s) = %s, delta = %s", thisX, thisPoint, thisPoint.point.getMean() - initialPoint.point.getMean()));
        if (loops++ > 100) {
          monitor.log(String.format("Loops = %s", loops));
          return this;
        }
      }
    }
  }
}
