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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.layers.Delta;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.SimpleLineSearchCursor;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.Util;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Layer rate diagnostic trainer.
 */
public class LayerRateDiagnosticTrainer {
  
  
  /**
   * The type Layer stats.
   */
  public static class LayerStats {
    /**
     * The Rate.
     */
    public final double rate;
    /**
     * The Delta.
     */
    public final double delta;
  
    /**
     * Instantiates a new Layer stats.
     *
     * @param rate  the rate
     * @param delta the delta
     */
    public LayerStats(double rate, double delta) {
      this.rate = rate;
      this.delta = delta;
    }
  
    @Override
    public String toString() {
      final StringBuffer sb = new StringBuffer("{");
      sb.append("rate=").append(rate);
      sb.append(", delta=").append(delta);
      sb.append('}');
      return sb.toString();
    }
  }
  
  private final Trainable subject;
  private OrientationStrategy orientation;
  private Duration timeout;
  private double terminateThreshold;
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  private boolean strict = false;
  private final Map<NNLayer,LayerStats> layerRates = new HashMap<>();
  
  /**
   * Instantiates a new Layer rate diagnostic trainer.
   *
   * @param subject the subject
   */
  public LayerRateDiagnosticTrainer(Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
    this.setOrientation(new GradientDescent());
  }
  
  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }
  
  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   * @return the max iterations
   */
  public LayerRateDiagnosticTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  /**
   * Run map.
   *
   * @return the map
   */
  public Map<NNLayer, LayerStats> run() {
    long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
    PointSample measure = measure();
    ArrayList<NNLayer> layers = new ArrayList<>(measure.weights.map.keySet());
    mainLoop: while (timeoutMs > System.currentTimeMillis() && measure.value > terminateThreshold) {
      if(currentIteration.get() > maxIterations) break;
      final PointSample initialPhasePoint = measure();
      
      measure = initialPhasePoint;
      subiterationLoop: for(int subiteration = 0; subiteration<iterationsPerSample; subiteration++) {
        if(currentIteration.incrementAndGet() > maxIterations) break;
        
        {
          SimpleLineSearchCursor orient = (SimpleLineSearchCursor) getOrientation().orient(subject, measure, monitor);
          double stepSize = 1e-12 * orient.origin.value;
          DeltaSet pointB = orient.step(stepSize, monitor).point.delta.copy();
          DeltaSet pointA = orient.step(0.0, monitor).point.delta.copy();
          DeltaSet d1 = pointA;
          DeltaSet d2 = d1.add(pointB.scale(-1)).scale(1.0 / stepSize);
          Map<NNLayer,Double> steps = new HashMap<>();
          double overallStepEstimate = d1.getMagnitude() / d2.getMagnitude();
          for(NNLayer layer : layers) {
            Delta a = d2.get(layer, (double[]) null);
            Delta b = d1.get(layer, (double[]) null);
            double bmag = Math.sqrt(b.sumSq());
            double amag = Math.sqrt(a.sumSq());
            double dot = a.dot(b) / (amag * bmag);
            double idealSize = bmag / (amag * dot);
            steps.put(layer, idealSize);
            monitor.log(String.format("Layers stats: %s (%s, %s, %s) => %s", layer, amag, bmag, dot, idealSize));
          }
          monitor.log(String.format("Estimated ideal rates for layers: %s (%s overall; probed at %s)", steps, overallStepEstimate, stepSize));
        }
  
  
        SimpleLineSearchCursor bestOrient = null;
        PointSample bestPoint = null;
        layerLoop: for(NNLayer layer : layers) {
          SimpleLineSearchCursor orient = (SimpleLineSearchCursor) getOrientation().orient(subject, measure, monitor);
          DeltaSet direction = filterDirection(orient.direction, layer);
          if(direction.getMagnitude() == 0) {
            monitor.log(String.format("Zero derivative for layer %s; skipping", layer));
            continue layerLoop;
          }
          orient = new SimpleLineSearchCursor(orient.subject, orient.origin, direction);
          PointSample previous = measure;
          measure = getLineSearchStrategy().step(orient, monitor);
          if(isStrict()) {
            monitor.log(String.format("Iteration %s reverting. Error: %s", currentIteration.get(), measure.value));
            monitor.log(String.format("Optimal rate for layer %s: %s", layer.getName(), measure.getRate()));
            if(null == bestPoint || bestPoint.value < measure.value) {
              bestOrient = orient;
              bestPoint = measure;
            }
            getLayerRates().put(layer, new LayerStats(measure.getRate(),initialPhasePoint.value-measure.value));
            orient.step(0, monitor);
            measure = previous;
          } else if(previous.value == measure.value) {
            monitor.log(String.format("Iteration %s failed. Error: %s", currentIteration.get(), measure.value));
          } else {
            monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), measure.value));
            monitor.log(String.format("Optimal rate for layer %s: %s", layer.getName(), measure.getRate()));
            getLayerRates().put(layer, new LayerStats(measure.getRate(),initialPhasePoint.value-measure.value));
          }
        }
        monitor.log(String.format("Ideal rates: %s", getLayerRates()));
        if(null != bestPoint) {
          bestOrient.step(bestPoint.rate, monitor);
        }
        monitor.onStepComplete(new Step(measure, currentIteration.get()));
      }
    }
    return getLayerRates();
  }
  
  /**
   * Gets line search strategy.
   *
   * @return the line search strategy
   */
  protected LineSearchStrategy getLineSearchStrategy() {
    return new QuadraticSearch();
  }
  
  private DeltaSet filterDirection(DeltaSet direction, NNLayer layer) {
    DeltaSet maskedDelta = new DeltaSet();
    direction.map.forEach((layer2,delta)->maskedDelta.get(layer2,delta.target));
    maskedDelta.get(layer,layer.state().get(0)).accumulate(direction.get(layer, (double[]) null).getDelta());
    return maskedDelta;
  }
  
  /**
   * Measure point sample.
   *
   * @return the point sample
   */
  public PointSample measure() {
    PointSample currentPoint;
    int retries = 0;
    do {
      if(!subject.resetSampling() && retries>0) throw new RuntimeException();
      if (10 < retries++) throw new RuntimeException();
      currentPoint = subject.measure();
    } while (!Double.isFinite(currentPoint.value));
    assert (Double.isFinite(currentPoint.value));
    return currentPoint;
  }
  
  /**
   * Gets timeout.
   *
   * @return the timeout
   */
  public Duration getTimeout() {
    return timeout;
  }
  
  /**
   * Sets timeout.
   *
   * @param timeout the timeout
   * @return the timeout
   */
  public LayerRateDiagnosticTrainer setTimeout(Duration timeout) {
    this.timeout = timeout;
    return this;
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public LayerRateDiagnosticTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public LayerRateDiagnosticTrainer setTimeout(int number, TemporalUnit units) {
    this.timeout = Duration.of(number, units);
    return this;
  }
  
  /**
   * Gets terminate threshold.
   *
   * @return the terminate threshold
   */
  public double getTerminateThreshold() {
    return terminateThreshold;
  }
  
  /**
   * Sets terminate threshold.
   *
   * @param terminateThreshold the terminate threshold
   * @return the terminate threshold
   */
  public LayerRateDiagnosticTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }
  
  /**
   * Gets monitor.
   *
   * @return the monitor
   */
  public TrainingMonitor getMonitor() {
    return monitor;
  }
  
  /**
   * Sets monitor.
   *
   * @param monitor the monitor
   * @return the monitor
   */
  public LayerRateDiagnosticTrainer setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
    return this;
  }
  
  /**
   * Gets current iteration.
   *
   * @return the current iteration
   */
  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }
  
  /**
   * Sets current iteration.
   *
   * @param currentIteration the current iteration
   * @return the current iteration
   */
  public LayerRateDiagnosticTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  /**
   * Gets iterations per sample.
   *
   * @return the iterations per sample
   */
  public int getIterationsPerSample() {
    return iterationsPerSample;
  }
  
  /**
   * Sets iterations per sample.
   *
   * @param iterationsPerSample the iterations per sample
   * @return the iterations per sample
   */
  public LayerRateDiagnosticTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
  /**
   * Is strict boolean.
   *
   * @return the boolean
   */
  public boolean isStrict() {
    return strict;
  }
  
  /**
   * Sets strict.
   *
   * @param strict the strict
   * @return the strict
   */
  public LayerRateDiagnosticTrainer setStrict(boolean strict) {
    this.strict = strict;
    return this;
  }
  
  /**
   * Gets orientation.
   *
   * @return the orientation
   */
  public OrientationStrategy getOrientation() {
    return orientation;
  }
  
  /**
   * Sets orientation.
   *
   * @param orientation the orientation
   * @return the orientation
   */
  public LayerRateDiagnosticTrainer setOrientation(OrientationStrategy orientation) {
    this.orientation = orientation;
    return this;
  }
  
  /**
   * Gets layer rates.
   *
   * @return the layer rates
   */
  public Map<NNLayer, LayerStats> getLayerRates() {
    return layerRates;
  }
}
