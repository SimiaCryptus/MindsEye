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

public class SimpleLearningRate implements LineSearchStrategy {
  
  private double rate = 1e-4;
  private double rateGrowth = 1.0;
  private double rateShrink = 0.5;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    rate *= rateGrowth; // Keep memory of rate from one iteration to next, but have a bias for growing the value
    LineSearchPoint startPoint = cursor.step(0, monitor);
    double startLineDeriv = startPoint.derivative; // theta'(0)
    double startValue = startPoint.point.value; // theta(0)
    LineSearchPoint lastStep = null;
    while (true) {
      double lastValue = (null == lastStep)?Double.POSITIVE_INFINITY:lastStep.point.value;
      if(!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      lastStep = cursor.step(rate, monitor);
      lastValue = lastStep.point.value;
      if(!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      if (lastValue > startValue) {
        monitor.log(String.format("Non-decreasing step. %s > %s New rate: " + rate, lastValue, startValue));
        rate *= rateShrink;
      } else {
        return lastStep.point;
      }
    }
  }
  
  public double getRate() {
    return rate;
  }
  
  public SimpleLearningRate setRate(double rate) {
    this.rate = rate;
    return this;
  }
  
  public double getRateGrowth() {
    return rateGrowth;
  }
  
  public SimpleLearningRate setRateGrowth(double rateGrowth) {
    this.rateGrowth = rateGrowth;
    return this;
  }
  
  public double getRateShrink() {
    return rateShrink;
  }
  
  public SimpleLearningRate setRateShrink(double rateShrink) {
    this.rateShrink = rateShrink;
    return this;
  }
}
