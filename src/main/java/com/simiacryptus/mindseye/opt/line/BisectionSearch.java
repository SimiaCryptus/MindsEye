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
import com.simiacryptus.mindseye.eval.Trainable.PointSample;

/**
 * The type Bisection search.
 */
public class BisectionSearch implements LineSearchStrategy {
  
  private double zeroTol = 1e-20;
  private double currentRate = 1.0;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    
    double leftX = 0;
    double leftLineDeriv;
    double leftValue;
    {
      LineSearchPoint searchPoint = cursor.step(leftX, monitor);
      monitor.log(String.format("F(%s) = %s", leftX, searchPoint));
      leftLineDeriv = searchPoint.derivative;
      leftValue = searchPoint.point.value;
    }
    
    double rightRight = Double.POSITIVE_INFINITY;
    double rightX;
    double rightLineDeriv;
    double rightValue;
    double rightRightSoft = this.currentRate * 2;
    LineSearchPoint rightPoint;
    int loopCount = 0;
    while (true) {
      rightX = (leftX + Math.min(rightRight, rightRightSoft)) / 2;
      rightPoint = cursor.step(rightX, monitor);
      monitor.log(String.format("F(%s)@%s = %s", rightX, loopCount, rightPoint));
      rightLineDeriv = rightPoint.derivative;
      rightValue = rightPoint.point.value;
      if (loopCount++ > 100) break;
      if ((rightRight - leftX) * 2.0 / (leftX + rightRight) < Math.pow(10, -3)) {
        monitor.log(String.format("Right limit is nonconvergent at %s/%s", leftX, rightRight));
        return cursor.step(leftX, monitor).point;
      }
      if (rightValue > leftValue) {
        rightRight = rightX;
        monitor.log(String.format("Right is at most %s", rightX));
      }
      else if (rightLineDeriv < 0) {
        rightRightSoft *= 2.0;
        leftLineDeriv = rightLineDeriv;
        leftValue = rightValue;
        leftX = rightX;
        monitor.log(String.format("Right is at least %s", rightX));
      }
      else {
        break;
      }
    }
    
    if (this.currentRate < rightX) {
      currentRate = rightX;
      return rightPoint.point;
    }
    
    LineSearchPoint searchPoint;
    loopCount = 0;
    while (true) {
      double thisX;
      thisX = (rightX + leftX) / 2;
      searchPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s", thisX, searchPoint));
      if (loopCount++ > 1000) return searchPoint.point;
      if (searchPoint.derivative < -zeroTol) {
        if (leftX == thisX) {
          monitor.log(String.format("End (static left) at %s", thisX));
          currentRate = thisX;
          return searchPoint.point;
        }
        leftLineDeriv = searchPoint.derivative;
        leftValue = searchPoint.point.value;
        leftX = thisX;
      }
      else if (searchPoint.derivative > zeroTol) {
        if (rightX == thisX) {
          monitor.log(String.format("End (static right) at %s", thisX));
          currentRate = thisX;
          return searchPoint.point;
        }
        rightLineDeriv = searchPoint.derivative;
        rightValue = searchPoint.point.value;
        rightX = thisX;
      }
      else {
        monitor.log(String.format("End (at zero) at %s", thisX));
        currentRate = thisX;
        return searchPoint.point;
      }
      if (Math.log10((rightX - leftX) * 2.0 / (leftX + rightX)) < -1) {
        monitor.log(String.format("End (narrow range) at %s to %s", rightX, leftX));
        currentRate = thisX;
        return searchPoint.point;
      }
    }
  }
  
  /**
   * Gets zero tol.
   *
   * @return the zero tol
   */
  public double getZeroTol() {
    return zeroTol;
  }
  
  /**
   * Sets zero tol.
   *
   * @param zeroTol the zero tol
   * @return the zero tol
   */
  public BisectionSearch setZeroTol(double zeroTol) {
    this.zeroTol = zeroTol;
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
  public BisectionSearch setCurrentRate(double currentRate) {
    this.currentRate = currentRate;
    return this;
  }
}
