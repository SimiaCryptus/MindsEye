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
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;

public class RoundRobinTrainer {
  
  
  private final Trainable subject;
  private Duration timeout;
  private double terminateThreshold;
  private List<OrientationStrategy> orientations = new ArrayList<>(Arrays.asList(new LBFGS()));
  private Function<String, LineSearchStrategy> lineSearchFactory = s -> new ArmijoWolfeConditions();
  private Map<String,LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  
  public RoundRobinTrainer(Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
  }
  
  public int getMaxIterations() {
    return maxIterations;
  }
  
  public RoundRobinTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  public double run() {
    long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
    PointSample currentPoint = measure();
    mainLoop: while (timeoutMs > System.currentTimeMillis() && currentPoint.value > terminateThreshold) {
      if(currentIteration.get() > maxIterations) break;
      currentPoint = measure();
      subiterationLoop: for(int subiteration = 0; subiteration<iterationsPerSample; subiteration++) {
        if(currentIteration.incrementAndGet() > maxIterations) break;
        for(OrientationStrategy orientation : orientations) {
          LineSearchCursor direction = orientation.orient(subject, currentPoint, monitor);
          String directionType = direction.getDirectionType() + "+" + Long.toHexString(System.identityHashCode(orientation));
          LineSearchStrategy lineSearchStrategy;
          if(lineSearchStrategyMap.containsKey(directionType)) {
            lineSearchStrategy = lineSearchStrategyMap.get(directionType);
          } else {
            System.out.println(String.format("Constructing line search parameters: %s", directionType));
            lineSearchStrategy = lineSearchFactory.apply(directionType);
            lineSearchStrategyMap.put(directionType, lineSearchStrategy);
          }
          PointSample previous = currentPoint;
          currentPoint = lineSearchStrategy.step(direction, monitor);
          monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
          if(previous.value == currentPoint.value) {
            if(subject.resetSampling()) {
              monitor.log(String.format("Iteration %s failed, retrying. Error: %s", currentIteration.get(), currentPoint.value));
              break subiterationLoop;
            } else {
              monitor.log(String.format("Iteration %s failed, aborting. Error: %s", currentIteration.get(), currentPoint.value));
              break mainLoop;
            }
          } else {
            monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), currentPoint.value));
          }
        }
      }
    }
    return null == currentPoint ? Double.NaN : currentPoint.value;
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
  
  public RoundRobinTrainer setTimeout(Duration timeout) {
    this.timeout = timeout;
    return this;
  }
  
  public RoundRobinTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  public RoundRobinTrainer setTimeout(int number, TemporalUnit units) {
    this.timeout = Duration.of(number, units);
    return this;
  }
  
  public double getTerminateThreshold() {
    return terminateThreshold;
  }
  
  public RoundRobinTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }
  
  public List<OrientationStrategy> getOrientations() {
    return orientations;
  }
  
  public RoundRobinTrainer setOrientations(OrientationStrategy... orientations) {
    this.orientations = new ArrayList<>(Arrays.asList(orientations));
    return this;
  }
  
  public TrainingMonitor getMonitor() {
    return monitor;
  }
  
  public RoundRobinTrainer setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
    return this;
  }
  
  public AtomicInteger getCurrentIteration() {
    return currentIteration;
  }
  
  public RoundRobinTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  public Function<String, LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }
  
  public RoundRobinTrainer setLineSearchFactory(Supplier<LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = s -> lineSearchFactory.get();
    return this;
  }
  
  public RoundRobinTrainer setLineSearchFactory(Function<String, LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
    return this;
  }
  
  public int getIterationsPerSample() {
    return iterationsPerSample;
  }
  
  public RoundRobinTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
}
