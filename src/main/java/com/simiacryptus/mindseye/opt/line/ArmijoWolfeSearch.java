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

import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

/**
 * Commonly used "loose" criteria for the line search iteration.
 */
public class ArmijoWolfeSearch implements LineSearchStrategy {
  
  private double minAlpha = 1e-15;
  private double maxAlpha = 1e4;
  private double c1 = 1e-6;
  private double c2 = 0.9;
  private double alpha = 1.0;
  private double alphaGrowth = Math.pow(10.0, Math.pow(3.0, -1.0));
  private boolean strongWolfe = true;
  private double absoluteTolerance = 1e-15;
  private double relativeTolerance = 1e-2;
  
  @Override
  public PointSample step(LineSearchCursor cursor, TrainingMonitor monitor) {
    alpha *= alphaGrowth; // Keep memory of alpha from one iteration to next, but have a bias for growing the value
    double mu = 0;
    double nu = Double.POSITIVE_INFINITY;
    final LineSearchPoint startPoint = cursor.step(0, monitor);
    final double startLineDeriv = startPoint.derivative; // theta'(0)
    final double startValue = startPoint.point.getMean(); // theta(0)
    if (0 <= startPoint.derivative) {
      monitor.log(String.format("th(0)=%s;dx=%s (ERROR: Starting derivative negative)", startValue, startLineDeriv));
      return cursor.step(0, monitor).point;
    }
    monitor.log(String.format("th(0)=%s;dx=%s", startValue, startLineDeriv));
    LineSearchPoint lastStep = null;
    int stepBias = 0;
    double bestAlpha = 0;
    double bestValue = startPoint.point.getMean();
    while (true) {
      if (!isAlphaValid()) {
        monitor.log(String.format("INVALID ALPHA: th(0)=%s;th'(0)=%s;", startValue, startLineDeriv));
        return cursor.step(bestAlpha, monitor).point;
      }
      double lastValue = (null == lastStep) ? Double.POSITIVE_INFINITY : lastStep.point.getMean();
      if (!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      if (mu >= nu - absoluteTolerance) {
        monitor.log(String.format("mu >= nu: th(0)=%s;th'(0)=%s;", startValue, startLineDeriv));
        loosenMetaparameters();
        return cursor.step(bestAlpha, monitor).point;
      }
      
      if ((nu - mu) < nu * relativeTolerance) {
        monitor.log(String.format("mu /= nu: th(0)=%s;th'(0)=%s;", startValue, startLineDeriv));
        loosenMetaparameters();
        return cursor.step(bestAlpha, monitor).point;
      }
      if (Math.abs(alpha) < minAlpha) {
        alpha = 1;
        monitor.log(String.format("MIN ALPHA: th(0)=%s;th'(0)=%s;", startValue, startLineDeriv));
        return cursor.step(bestAlpha, monitor).point;
      }
      if (Math.abs(alpha) > maxAlpha) {
        alpha = 1;
        monitor.log(String.format("MAX ALPHA: th(0)=%s;th'(0)=%s;", startValue, startLineDeriv));
        return cursor.step(bestAlpha, monitor).point;
      }
      lastStep = cursor.step(alpha, monitor);
      lastValue = lastStep.point.getMean();
      if (bestValue > lastValue) {
        bestAlpha = alpha;
        bestValue = lastValue;
      }
      if (!Double.isFinite(lastValue)) lastValue = Double.POSITIVE_INFINITY;
      if (lastValue > startValue + alpha * c1 * startLineDeriv) {
        // Value did not decrease (enough) - It is gauranteed to decrease given an infitefimal rate; the rate must be less than this; this is a new ceiling
        monitor.log(String.format("Armijo: th(%s)=%s; dx=%s delta=%s", alpha, lastValue, lastStep.derivative, startValue - lastValue));
        nu = alpha;
        stepBias = Math.min(-1, stepBias - 1);
      }
      else if (isStrongWolfe() && lastStep.derivative > 0) {
        // If the slope is increasing, then we can go lower by choosing a lower rate; this is a new ceiling
        monitor.log(String.format("WOLF (strong): th(%s)=%s; dx=%s delta=%s", alpha, lastValue, lastStep.derivative, startValue - lastValue));
        nu = alpha;
        stepBias = Math.min(-1, stepBias - 1);
      }
      else if (lastStep.derivative < c2 * startLineDeriv) {
        // Current slope decreases at no more than X - If it is still decreasing that fast, we know we want a rate of least this value; this is a new floor
        monitor.log(String.format("WOLFE (weak): th(%s)=%s; dx=%s delta=%s", alpha, lastValue, lastStep.derivative, startValue - lastValue));
        mu = alpha;
        stepBias = Math.max(1, stepBias + 1);
      }
      else {
        monitor.log(String.format("END: th(%s)=%s; dx=%s delta=%s", alpha, lastValue, lastStep.derivative, startValue - lastValue));
        stepBias = 0;
        return lastStep.point;
      }
      if (!Double.isFinite(nu)) {
        alpha = (1 + Math.abs(stepBias)) * alpha;
      }
      else if (0.0 == mu) {
        alpha = nu / (1 + Math.abs(stepBias));
      }
      else {
        alpha = (mu + nu) / 2;
      }
    }
  }
  
  /**
   * Loosen metaparameters.
   */
  public void loosenMetaparameters() {
    c1 *= 0.2;
    c2 = Math.pow(c2, c2 < 1 ? 1.5 : (1 / 1.5));
    strongWolfe = false;
  }
  
  private boolean isAlphaValid() {
    return Double.isFinite(alpha) && (0 <= alpha);
  }
  
  /**
   * Gets alpha growth.
   *
   * @return the alpha growth
   */
  public double getAlphaGrowth() {
    return alphaGrowth;
  }
  
  /**
   * Sets alpha growth.
   *
   * @param alphaGrowth the alpha growth
   * @return the alpha growth
   */
  public ArmijoWolfeSearch setAlphaGrowth(double alphaGrowth) {
    this.alphaGrowth = alphaGrowth;
    return this;
  }
  
  /**
   * Gets c 1.
   *
   * @return the c 1
   */
  public double getC1() {
    return c1;
  }
  
  /**
   * Sets c 1.
   *
   * @param c1 the c 1
   * @return the c 1
   */
  public ArmijoWolfeSearch setC1(double c1) {
    this.c1 = c1;
    return this;
  }
  
  /**
   * Gets c 2.
   *
   * @return the c 2
   */
  public double getC2() {
    return c2;
  }
  
  /**
   * Sets c 2.
   *
   * @param c2 the c 2
   * @return the c 2
   */
  public ArmijoWolfeSearch setC2(double c2) {
    this.c2 = c2;
    return this;
  }
  
  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }
  
  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   * @return the alpha
   */
  public ArmijoWolfeSearch setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
  
  /**
   * Gets min alpha.
   *
   * @return the min alpha
   */
  public double getMinAlpha() {
    return minAlpha;
  }
  
  /**
   * Sets min alpha.
   *
   * @param minAlpha the min alpha
   * @return the min alpha
   */
  public ArmijoWolfeSearch setMinAlpha(double minAlpha) {
    this.minAlpha = minAlpha;
    return this;
  }
  
  /**
   * Gets max alpha.
   *
   * @return the max alpha
   */
  public double getMaxAlpha() {
    return maxAlpha;
  }
  
  /**
   * Sets max alpha.
   *
   * @param maxAlpha the max alpha
   * @return the max alpha
   */
  public ArmijoWolfeSearch setMaxAlpha(double maxAlpha) {
    this.maxAlpha = maxAlpha;
    return this;
  }
  
  /**
   * Is strong wolfe boolean.
   *
   * @return the boolean
   */
  public boolean isStrongWolfe() {
    return strongWolfe;
  }
  
  /**
   * Sets strong wolfe.
   *
   * @param strongWolfe the strong wolfe
   * @return the strong wolfe
   */
  public ArmijoWolfeSearch setStrongWolfe(boolean strongWolfe) {
    this.strongWolfe = strongWolfe;
    return this;
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
  public ArmijoWolfeSearch setAbsoluteTolerance(double absoluteTolerance) {
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
  public ArmijoWolfeSearch setRelativeTolerance(double relativeTolerance) {
    this.relativeTolerance = relativeTolerance;
    return this;
  }
}
