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
  
  private double absoluteTolerance = 1e-20;
  private double relativeTolerance = 1e-4;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    
    double leftX = 0;
    LineSearchPoint leftPoint = cursor.step(leftX, monitor);
    monitor.log(String.format("F(%s) = %s", leftX, leftPoint));
  
    double rightX = Math.abs(leftPoint.point.value * 1e-4 / leftPoint.derivative);
    LineSearchPoint rightPoint = cursor.step(rightX, monitor);
    monitor.log(String.format("F(%s) = %s", rightX, rightPoint));
    
    while(true) {
      double a = (rightPoint.derivative - leftPoint.derivative) / (rightX - leftX);
      double b = rightPoint.derivative - a * rightX;
      double thisX = - b / a;
      LineSearchPoint thisPoint = cursor.step(thisX, monitor);
      monitor.log(String.format("F(%s) = %s", thisX, thisPoint));
      if(Math.abs(thisPoint.derivative) < absoluteTolerance) {
        return thisPoint.point;
      }
      boolean sameSign = leftPoint.derivative * rightPoint.derivative > 0 ? true : false;
      boolean test;
      if(sameSign) {
        test = Math.abs(rightPoint.derivative - thisPoint.derivative) > Math.abs(leftPoint.derivative - thisPoint.derivative);
      } else {
        test = thisPoint.derivative > 0;
      }
      if(test) {
        rightX = thisX;
        rightPoint = thisPoint;
        monitor.log(String.format("Right bracket at %s", thisX));
      } else {
        leftPoint = thisPoint;
        leftX = thisX;
        monitor.log(String.format("Left bracket at %s", thisX));
      }
      if(Math.abs((rightX - leftX) * 2.0 / (leftX + rightX)) < getRelativeTolerance()) {
        monitor.log(String.format("Interval converged at %s/%s", leftX, rightX));
        return cursor.step(leftX, monitor).point.setRate(leftX);
      }
    }
  
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
