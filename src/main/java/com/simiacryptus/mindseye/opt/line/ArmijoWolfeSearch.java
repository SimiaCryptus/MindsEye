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

public class ArmijoWolfeSearch implements LineSearchStrategy {
  
  private double minAlpha = 1e-15;
  private double maxAlpha = 1e2;
  private double c1 = 1e-6;
  private double c2 = 0.9;
  private double alpha = 1.0;
  private double alphaGrowth = Math.pow(10.0, Math.pow(3.0, -1.0));
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    alpha *= alphaGrowth; // Keep memory of alpha from one iteration to next, but have a bias for growing the value
    // See http://cs.nyu.edu/overton/mstheses/skajaa/msthesis.pdf page 14
    double mu = 0;
    double nu = Double.POSITIVE_INFINITY;
    LineSearchPoint startPoint = cursor.step(0, monitor);
    double startLineDeriv = startPoint.derivative; // theta'(0)
    double startValue = startPoint.point.value; // theta(0)
    LineSearchPoint lastStep = null;
    int stepBias = 0;
    while (true) {
      if (!isAlphaValid()) {
        monitor.log(String.format("INVALID ALPHA: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
        return cursor.step(0, monitor).point;
      }
      double lastValue = (null == lastStep)?Double.POSITIVE_INFINITY:lastStep.point.value;
      if(!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      if (mu >= nu) {
        monitor.log(String.format("mu >= nu: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
        loosenMetaparameters();
        if(null != lastStep && lastValue < startValue) return lastStep.point;
        return cursor.step(0, monitor).point;
      }
      if ((nu / mu) < (11.0 / 10.0)) {
        monitor.log(String.format("mu >= nu: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
        loosenMetaparameters();
        if(null != lastStep && lastValue < startValue) return lastStep.point;
        return cursor.step(0, monitor).point;
      }
      if (Math.abs(alpha) < minAlpha) {
        alpha = 1;
        monitor.log(String.format("MIN ALPHA: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
        if(null != lastStep && lastValue < startValue) return lastStep.point;
        return cursor.step(0, monitor).point;
      }
      if (Math.abs(alpha) > maxAlpha) {
        alpha = 1;
        monitor.log(String.format("MAX ALPHA: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s", startValue, startLineDeriv, mu, alpha, nu));
        if(null != lastStep && lastValue < startValue) return lastStep.point;
        return cursor.step(0, monitor).point;
      }
      lastStep = cursor.step(alpha, monitor);
      lastValue = lastStep.point.value;
      if(!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      if (lastValue > startValue + alpha * c1 * startLineDeriv) {
        // Armijo condition fails
        monitor.log(String.format("ARMIJO: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f > %f;th'(alpha)=%f >= %f",
            startValue, startLineDeriv, mu, alpha, nu, lastValue, startValue + alpha * c1 * startLineDeriv, lastStep.derivative, c2 * startLineDeriv));
        nu = alpha;
        stepBias = Math.min(-1, stepBias-1);
      } else if (lastStep.derivative < c2 * startLineDeriv) {
        // Weak Wolfe condition fails
        monitor.log(String.format("WOLFE: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%f <= %f;th'(alpha)=%f < %f",
            startValue, startLineDeriv, mu, alpha, nu, lastValue, startValue + alpha * c1 * startLineDeriv, lastStep.derivative, c2 * startLineDeriv));
        mu = alpha;
        stepBias = Math.max(1, stepBias+1);
      } else {
        monitor.log(String.format("END: th(0)=%5f;th'(0)=%5f;\t%s - %s - %s\tth(alpha)=%5f;th'(alpha)=%5f",
            startValue, startLineDeriv, mu, alpha, nu, lastValue, lastStep.derivative));
        stepBias = 0;
        return lastStep.point;
      }
      if (!Double.isFinite(nu)) {
        alpha = (1+Math.abs(stepBias)) * alpha;
      } else if (0.0 == mu) {
        alpha = nu / (1+Math.abs(stepBias));
      } else {
        alpha = (mu + nu) / 2;
      }
    }
  }
  
  public void loosenMetaparameters() {
    c1 *= 0.2;
    c2 = Math.pow(c2,c2<1?0.3:3);
  }
  
  private boolean isAlphaValid() {
    return Double.isFinite(alpha) && (0 <= alpha);
  }
  
  public double getAlphaGrowth() {
    return alphaGrowth;
  }
  
  public ArmijoWolfeSearch setAlphaGrowth(double alphaGrowth) {
    this.alphaGrowth = alphaGrowth;
    return this;
  }
  
  public double getC1() {
    return c1;
  }
  
  public ArmijoWolfeSearch setC1(double c1) {
    this.c1 = c1;
    return this;
  }
  
  public double getC2() {
    return c2;
  }
  
  public ArmijoWolfeSearch setC2(double c2) {
    this.c2 = c2;
    return this;
  }
  
  public double getAlpha() {
    return alpha;
  }
  
  public ArmijoWolfeSearch setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
  
  public double getMinAlpha() {
    return minAlpha;
  }
  
  public ArmijoWolfeSearch setMinAlpha(double minAlpha) {
    this.minAlpha = minAlpha;
    return this;
  }
  
  public double getMaxAlpha() {
    return maxAlpha;
  }
  
  public ArmijoWolfeSearch setMaxAlpha(double maxAlpha) {
    this.maxAlpha = maxAlpha;
    return this;
  }
}
