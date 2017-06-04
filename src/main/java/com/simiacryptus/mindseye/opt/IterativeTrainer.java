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

import com.simiacryptus.mindseye.opt.line.ArmijoWolfeConditions;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable.PointSample;
import com.simiacryptus.util.Util;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

public class IterativeTrainer {
  
  
  private final Trainable subject;
  private Duration timeout;
  private double terminateThreshold;
  private OrientationStrategy orientation = new LBFGS();
  private Supplier<LineSearchStrategy> lineSearchFactory = () -> new ArmijoWolfeConditions();
  private Map<String,LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  
  public IterativeTrainer(Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
  }
  
  public int getMaxIterations() {
    return maxIterations;
  }
  
  public IterativeTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  public double run() {
    long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
    PointSample currentPoint = measure();
    while (timeoutMs > System.currentTimeMillis() && currentPoint.value > terminateThreshold && currentIteration.get() < maxIterations) {
      currentPoint = measure();
      subiterationLoop: for(int subiteration = 0; subiteration<iterationsPerSample; subiteration++) {
        LineSearchCursor direction = orientation.orient(subject, currentPoint, monitor);
        String directionType = direction.getDirectionType();
        LineSearchStrategy lineSearchStrategy = lineSearchStrategyMap.get(directionType);
        if(null == lineSearchStrategy) {
          System.out.println(String.format("Constructing line search parameters: %s", directionType));
          lineSearchStrategy = lineSearchFactory.get();
          lineSearchStrategyMap.put(directionType, lineSearchStrategy);
        }
        PointSample previous = currentPoint;
        currentPoint = lineSearchStrategy.step(direction, monitor);
        monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.incrementAndGet(), currentPoint.value));
        monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
        if(previous.value == currentPoint.value) break subiterationLoop;
      }
    }
    return null == currentPoint ? Double.NaN : currentPoint.value;
  }
  
  public PointSample measure() {
    PointSample currentPoint;
    int retries = 0;
    do {
      if (10 < retries++) throw new RuntimeException();
      subject.resetSampling();
      currentPoint = subject.measure();
    } while (!Double.isFinite(currentPoint.value));
    assert (Double.isFinite(currentPoint.value));
    return currentPoint;
  }
  
  public Duration getTimeout() {
    return timeout;
  }
  
  public IterativeTrainer setTimeout(Duration timeout) {
    this.timeout = timeout;
    return this;
  }
  
  public IterativeTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  public IterativeTrainer setTimeout(int number, TemporalUnit units) {
    this.timeout = Duration.of(number, units);
    return this;
  }
  
  public double getTerminateThreshold() {
    return terminateThreshold;
  }
  
  public IterativeTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }
  
  public OrientationStrategy getOrientation() {
    return orientation;
  }
  
  public IterativeTrainer setOrientation(OrientationStrategy orientation) {
    this.orientation = orientation;
    return this;
  }
  
  public TrainingMonitor getMonitor() {
    return monitor;
  }
  
  public IterativeTrainer setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
    return this;
  }
  
  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }
  
  public IterativeTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  public Supplier<LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }
  
  public IterativeTrainer setLineSearchFactory(Supplier<LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
    return this;
  }
  
  public int getIterationsPerSample() {
    return iterationsPerSample;
  }
  
  public IterativeTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
  public static class Step {
    public final PointSample point;
    public final long time = System.currentTimeMillis();
    public final long iteration;
    
    private Step(PointSample point, long iteration) {
      this.point = point;
      this.iteration = iteration;
    }
  }
}
