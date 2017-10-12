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

import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Quadratic search.
 */
public class QuadraticSearch implements LineSearchStrategy {
  
  private double absoluteTolerance = 1e-12;
  private double relativeTolerance = 1e-2;
  private double initialDerivFactor = 0.95;
  private double currentRate = 0.0;
  private double minRate = 1e-10;
  private int maxIterations = 100;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    double leftX = 0;
    if(currentRate < getMinRate()) currentRate = getMinRate();
    PointSample pointSample = _step(cursor, monitor, leftX);
    setCurrentRate(pointSample.rate);
    return pointSample;
  }
  
  /**
   * Step point sample.
   *
   * @param cursor  the cursor
   * @param monitor the monitor
   * @param leftX   the left x
   * @return the point sample
   */
  public PointSample _step(LineSearchCursor cursor, TrainingMonitor monitor, double leftX) {
    LineSearchPoint leftPoint = cursor.step(leftX, monitor);
    monitor.log(String.format("F(%s) = %s; F' = %s", leftX, leftPoint, leftPoint.derivative));
    if (0 == leftPoint.derivative) return leftPoint.point;
    
    LocateInitialRightPoint locateInitialRightPoint = new LocateInitialRightPoint(cursor, monitor, leftPoint).apply();
    LineSearchPoint rightPoint = locateInitialRightPoint.getRightPoint();
    double rightX = locateInitialRightPoint.getRightX();
    
    int loops = 0;
    while (true) {
      double a = (rightPoint.derivative - leftPoint.derivative) / (rightX - leftX);
      double b = rightPoint.derivative - a * rightX;
      double thisX = -b / a;
      boolean isBracketed = Math.signum(leftPoint.derivative) != Math.signum(rightPoint.derivative);
      if (!Double.isFinite(thisX) || isBracketed && (leftX > thisX || rightX < thisX)) {
        thisX = (rightX + leftX) / 2;
      }
      if (!isBracketed && thisX < 0) {
        thisX = rightX * 2;
      }
      if (isSame(leftX, thisX) || isSame(thisX, rightX)) {
        thisX = (rightX + leftX) / 2;
      }
      LineSearchPoint thisPoint = cursor.step(thisX, monitor);
      if (loops++ > 100) {
        monitor.log(String.format("Loops = %s", loops));
        return filter(cursor, thisPoint.point, monitor);
      }
      monitor.log(String.format("F(%s)@%s = %s", thisX, loops, thisPoint));
      if (isSame(leftX, rightX)) {
        monitor.log(String.format("%s ~= %s", leftX, rightX));
        return filter(cursor, thisPoint.point, monitor);
      }
      boolean test;
      if (!isBracketed) {
        test = Math.abs(rightPoint.point.rate - thisPoint.point.rate) > Math.abs(leftPoint.point.rate - thisPoint.point.rate);
      }
      else {
        test = thisPoint.derivative < 0;
      }
      if (test) {
        if (thisPoint.point.value > leftPoint.point.value) {
          monitor.log(String.format("%s > %s", thisPoint.point.value, leftPoint.point.value));
          return filter(cursor, leftPoint.point, monitor);
        }
        if (!isBracketed && leftPoint.point.value < rightPoint.point.value) {
          rightX = leftX;
          rightPoint = leftPoint;
        }
        leftPoint = thisPoint;
        leftX = thisX;
        monitor.log(String.format("Left bracket at %s", thisX));
      }
      else {
        if (thisPoint.point.value > rightPoint.point.value) {
          monitor.log(String.format("%s > %s", thisPoint.point.value, rightPoint.point.value));
          return filter(cursor, rightPoint.point, monitor);
        }
        if (!isBracketed && rightPoint.point.value < leftPoint.point.value) {
          leftX = rightX;
          leftPoint = rightPoint;
        }
        rightX = thisX;
        rightPoint = thisPoint;
        monitor.log(String.format("Right bracket at %s", thisX));
      }
    }
  }
  
  private double stepSize = 1.0;
  
  private PointSample filter(LineSearchCursor cursor, PointSample point, TrainingMonitor monitor) {
    if (stepSize == 1.0) return point;
    return cursor.step(point.rate * stepSize, monitor).point;
  }
  
  /**
   * Is same boolean.
   *
   * @param a the a
   * @param b the b
   * @return the boolean
   */
  protected boolean isSame(double a, double b) {
    double diff = Math.abs(a - b);
    double scale = Math.max(Math.abs(a), Math.abs(b));
    return diff < absoluteTolerance || diff < (scale * relativeTolerance);
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
   * Gets step size.
   *
   * @return the step size
   */
  public double getStepSize() {
    return stepSize;
  }
  
  /**
   * Sets step size.
   *
   * @param stepSize the step size
   * @return the step size
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
  
  public double getMinRate() {
    return minRate;
  }
  
  public void setMinRate(double minRate) {
    this.minRate = minRate;
  }
  
  private class LocateInitialRightPoint {
    private LineSearchCursor cursor;
    private TrainingMonitor monitor;
    private LineSearchPoint initialPoint;
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
      thisX = getCurrentRate() > 0 ? getCurrentRate() : Math.abs(leftPoint.point.value * 1e-4 / leftPoint.derivative);
      thisPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s, F' = %s", thisX, thisPoint, thisPoint.derivative));
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
      double lastX = thisX;
      int loops = 0;
      while (true) {
        if (thisPoint.point.value > initialPoint.point.value) {
          thisX = thisX / 3;
        }
        else {
          if (thisPoint.derivative < initialDerivFactor * thisPoint.derivative) {
            thisX = thisX * 2;
          }
          else {
            //monitor.log(String.format("%s <= %s", rightPoint.point.value, leftPoint.point.value));
            break;
          }
        }
        if (isSame(lastX, thisX)) {
          monitor.log(String.format("%s ~= %s", lastX, thisX));
          break;
        }
        lastX = thisX;
        thisPoint = cursor.step(thisX, monitor);
        monitor.log(String.format("F(%s) = %s", thisX, thisPoint));
        if (loops++ > 100) {
          monitor.log(String.format("Loops = %s", loops));
          return this;
        }
      }
      return this;
    }
  }
}
