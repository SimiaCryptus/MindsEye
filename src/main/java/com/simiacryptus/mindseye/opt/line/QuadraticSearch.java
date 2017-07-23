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

public class QuadraticSearch implements LineSearchStrategy {
  
  private double absoluteTolerance = 1e-8;
  private double relativeTolerance = 1e-2;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    
    double leftX = 0;
    LineSearchPoint leftPoint = cursor.step(leftX, monitor);
    monitor.log(String.format("F(%s) = %s", leftX, leftPoint));
  
    double rightX = Math.abs(leftPoint.point.value * 1e-4 / leftPoint.derivative);
    LineSearchPoint rightPoint = cursor.step(rightX, monitor);
    monitor.log(String.format("F(%s) = %s", rightX, rightPoint));
  
  
    double lastX = rightX;
    while(true) {
      double startX = rightX;
      if (rightPoint.point.value > leftPoint.point.value) {
        rightX = rightX / 2;
      } else {
        break;
      }
      if(isSame(lastX, rightX)) break;
      lastX = startX;
      rightPoint = cursor.step(rightX, monitor);
      monitor.log(String.format("F(%s) = %s", rightX, rightPoint));
    }
    while(true) {
      double a = (rightPoint.derivative - leftPoint.derivative) / (rightX - leftX);
      double b = rightPoint.derivative - a * rightX;
      double thisX = - b / a;
      boolean isBracketed = Math.signum(leftPoint.derivative) != Math.signum(rightPoint.derivative);
      if(isBracketed && (leftX > thisX || rightX < thisX))
      {
        thisX = (rightX + leftX) / 2;
      }
      if(!isBracketed && thisX < 0) {
        thisX = rightX * 2;
      }
      if(isSame(leftX, thisX) || isSame(thisX, rightX))
      {
        thisX = (rightX + leftX) / 2;
      }
      LineSearchPoint thisPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s", thisX, thisPoint));
      if(isSame(leftX, rightX)) {
        return thisPoint.point;
      }
      boolean test;
      if(!isBracketed) {
        test = Math.abs(rightPoint.point.rate - thisPoint.point.rate) > Math.abs(leftPoint.point.rate - thisPoint.point.rate);
      } else {
        test = thisPoint.derivative < 0;
      }
      if(test) {
        if(thisPoint.point.value > leftPoint.point.value) return leftPoint.point;
        if(!isBracketed && leftPoint.point.value < rightPoint.point.value) {
          rightX = leftX;
          rightPoint = leftPoint;
        }
        leftPoint = thisPoint;
        leftX = thisX;
        monitor.log(String.format("Left bracket at %s", thisX));
      } else {
        if(thisPoint.point.value > rightPoint.point.value) return rightPoint.point;
        if(!isBracketed && rightPoint.point.value < leftPoint.point.value) {
          leftX = rightX;
          leftPoint = rightPoint;
        }
        rightX = thisX;
        rightPoint = thisPoint;
        monitor.log(String.format("Right bracket at %s", thisX));
      }
    }
  
  }
  
  protected boolean isSame(double a, double b) {
    double diff = Math.abs(a - b);
    double scale = Math.max(Math.abs(a), Math.abs(b));
    return diff < absoluteTolerance || diff < (scale * relativeTolerance);
  }
  
  public double getAbsoluteTolerance() {
    return absoluteTolerance;
  }
  
  public QuadraticSearch setAbsoluteTolerance(double absoluteTolerance) {
    this.absoluteTolerance = absoluteTolerance;
    return this;
  }
  
  public double getRelativeTolerance() {
    return relativeTolerance;
  }
  
  public QuadraticSearch setRelativeTolerance(double relativeTolerance) {
    this.relativeTolerance = relativeTolerance;
    return this;
  }
}
