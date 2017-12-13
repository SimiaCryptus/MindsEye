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

import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.util.Util;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * This basic training class provides support for a sequence of independant training configuration,
 * e.g. Generative Adversarial Networks.
 */
public class RoundRobinTrainer {
  
  
  private final Trainable subject;
  private final Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private Duration timeout;
  private double terminateThreshold;
  private List<? extends OrientationStrategy> orientations = new ArrayList<>(Arrays.asList(new LBFGS()));
  private Function<String, ? extends LineSearchStrategy> lineSearchFactory = s -> new ArmijoWolfeSearch();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  
  /**
   * Instantiates a new Round robin trainer.
   *
   * @param subject the subject
   */
  public RoundRobinTrainer(Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
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
  public RoundRobinTrainer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }
  
  /**
   * Run double.
   *
   * @return the double
   */
  public double run() {
    long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
    PointSample currentPoint = measure();
    mainLoop:
    while (timeoutMs > System.currentTimeMillis() && currentPoint.sum > terminateThreshold) {
      if (currentIteration.get() > maxIterations) break;
      currentPoint = measure();
      subiterationLoop:
      for (int subiteration = 0; subiteration < iterationsPerSample; subiteration++) {
        PointSample previousOrientations = currentPoint;
        orientationLoop:
        for (OrientationStrategy orientation : orientations) {
          if (currentIteration.incrementAndGet() > maxIterations) break;
          LineSearchCursor direction = orientation.orient(subject, currentPoint, monitor);
          String directionType = direction.getDirectionType() + "+" + Long.toHexString(System.identityHashCode(orientation));
          LineSearchStrategy lineSearchStrategy;
          if (lineSearchStrategyMap.containsKey(directionType)) {
            lineSearchStrategy = lineSearchStrategyMap.get(directionType);
          }
          else {
            System.out.println(String.format("Constructing line search parameters: %s", directionType));
            lineSearchStrategy = lineSearchFactory.apply(directionType);
            lineSearchStrategyMap.put(directionType, lineSearchStrategy);
          }
          PointSample previous = currentPoint;
          currentPoint = lineSearchStrategy.step(direction, monitor);
          monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
          if (previous.sum == currentPoint.sum) {
            monitor.log(String.format("Iteration %s failed, ignoring. Error: %s", currentIteration.get(), currentPoint.sum));
          }
          else {
            monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), currentPoint.sum));
          }
        }
        if (previousOrientations.sum <= currentPoint.sum) {
          if (subject.reseed(System.nanoTime())) {
            monitor.log(String.format("MacroIteration %s failed, retrying. Error: %s", currentIteration.get(), currentPoint.sum));
            break subiterationLoop;
          }
          else {
            monitor.log(String.format("MacroIteration %s failed, aborting. Error: %s", currentIteration.get(), currentPoint.sum));
            break mainLoop;
          }
        }
      }
    }
    return null == currentPoint ? Double.NaN : currentPoint.sum;
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
      if (!subject.reseed(System.nanoTime()) && retries > 0) throw new IterativeStopException();
      if (10 < retries++) throw new IterativeStopException();
      currentPoint = subject.measure(false, monitor);
    } while (!Double.isFinite(currentPoint.sum));
    assert (Double.isFinite(currentPoint.sum));
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
  public RoundRobinTrainer setTimeout(Duration timeout) {
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
  public RoundRobinTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public RoundRobinTrainer setTimeout(int number, TemporalUnit units) {
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
  public RoundRobinTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
  }
  
  /**
   * Gets orientations.
   *
   * @return the orientations
   */
  public List<? extends OrientationStrategy> getOrientations() {
    return orientations;
  }
  
  /**
   * Sets orientations.
   *
   * @param orientations the orientations
   * @return the orientations
   */
  public RoundRobinTrainer setOrientations(OrientationStrategy... orientations) {
    this.orientations = new ArrayList<>(Arrays.asList(orientations));
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
  public RoundRobinTrainer setMonitor(TrainingMonitor monitor) {
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
  public RoundRobinTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  /**
   * Gets line search factory.
   *
   * @return the line search factory
   */
  public Function<String, ? extends LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }
  
  /**
   * Sets line search factory.
   *
   * @param lineSearchFactory the line search factory
   * @return the line search factory
   */
  public RoundRobinTrainer setLineSearchFactory(Function<String, ? extends LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
    return this;
  }
  
  /**
   * Sets line search factory.
   *
   * @param lineSearchFactory the line search factory
   * @return the line search factory
   */
  public RoundRobinTrainer setLineSearchFactory(Supplier<LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = s -> lineSearchFactory.get();
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
  public RoundRobinTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
}
