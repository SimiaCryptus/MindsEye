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

import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.opt.line.*;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.Util;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class LayerRateDiagnosticTrainer {
  
  
  public static class LayerStats {
    public final double rate;
    public final double delta;
  
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
  
  public LayerRateDiagnosticTrainer(Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
    this.setOrientation(new GradientDescent());
  }
  
  public int getMaxIterations() {
    return maxIterations;
  }
  
  public LayerRateDiagnosticTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
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
            DeltaBuffer a = d2.get(layer, (double[]) null);
            DeltaBuffer b = d1.get(layer, (double[]) null);
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
  
  protected LineSearchStrategy getLineSearchStrategy() {
    return new QuadraticSearch();
  }
  
  private DeltaSet filterDirection(DeltaSet direction, NNLayer layer) {
    DeltaSet maskedDelta = new DeltaSet();
    direction.map.forEach((layer2,delta)->maskedDelta.get(layer2,delta.target));
    maskedDelta.get(layer,layer.state().get(0)).accumulate(direction.get(layer,(double[])null).delta);
    return maskedDelta;
  }
  
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
  
  public Duration getTimeout() {
    return timeout;
  }
  
  public LayerRateDiagnosticTrainer setTimeout(Duration timeout) {
    this.timeout = timeout;
    return this;
  }
  
  public LayerRateDiagnosticTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  public LayerRateDiagnosticTrainer setTimeout(int number, TemporalUnit units) {
    this.timeout = Duration.of(number, units);
    return this;
  }
  
  public double getTerminateThreshold() {
    return terminateThreshold;
  }
  
  public LayerRateDiagnosticTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }
  
  public TrainingMonitor getMonitor() {
    return monitor;
  }
  
  public LayerRateDiagnosticTrainer setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
    return this;
  }
  
  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }
  
  public LayerRateDiagnosticTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  public int getIterationsPerSample() {
    return iterationsPerSample;
  }
  
  public LayerRateDiagnosticTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
  public boolean isStrict() {
    return strict;
  }
  
  public LayerRateDiagnosticTrainer setStrict(boolean strict) {
    this.strict = strict;
    return this;
  }
  
  public OrientationStrategy getOrientation() {
    return orientation;
  }
  
  public LayerRateDiagnosticTrainer setOrientation(OrientationStrategy orientation) {
    this.orientation = orientation;
    return this;
  }
  
  public Map<NNLayer, LayerStats> getLayerRates() {
    return layerRates;
  }
}
