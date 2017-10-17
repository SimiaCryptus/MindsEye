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

import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.FailsafeLineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
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
import java.util.function.Function;

/**
 * The type Iterative trainer.
 */
public class ValidatingTrainer {
  
  private final Trainable subject;
  private Duration timeout;
  private double terminateThreshold;
  private OrientationStrategy orientation = new LBFGS();
  private Function<String, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
  private Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  
  /**
   * Instantiates a new Iterative trainer.
   *
   * @param subject the subject
   */
  public ValidatingTrainer(Trainable subject) {
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
  public ValidatingTrainer setMaxIterations(int maxIterations) {
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
    PointSample currentPoint = resetAndMeasure();
    while (timeoutMs > System.currentTimeMillis() && currentPoint.value > terminateThreshold) {
      if (shouldStop(timeoutMs)) break;
      if(!epoch(timeoutMs)) break;
      orientation.reset();
    }
    return null == currentPoint ? Double.NaN : currentPoint.value;
  }
  
  private boolean epoch(long timeoutMs) {
    PointSample currentPoint = resetAndMeasure();
    assert (0 < currentPoint.delta.map.size()) : "Nothing to optimize";
    for (int subiteration = 0; subiteration < iterationsPerSample || iterationsPerSample <= 0; subiteration++) {
      if (shouldStop(timeoutMs)) return false;
      TrainingStepResult epoch = step(currentPoint);
      currentPoint = epoch.getCurrentPoint();
      if (epoch.getPrevious().value <= epoch.getCurrentPoint().value) {
        if (subject.resetSampling()) {
          monitor.log(String.format("Iteration %s failed, retrying. Error: %s", currentIteration.get(), epoch.getCurrentPoint().value));
          return true;
        }
        else {
          monitor.log(String.format("Iteration %s failed, aborting. Error: %s", currentIteration.get(), epoch.getCurrentPoint().value));
          return false;
        }
      }
      else {
        monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), epoch.getCurrentPoint().value));
      }
      monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
    }
    return true;
  }
  
  public boolean shouldStop(long timeoutMs) {
    boolean stopTraining = timeoutMs < System.currentTimeMillis();
    stopTraining |= currentIteration.incrementAndGet() > maxIterations;
    return stopTraining;
  }
  
  private TrainingStepResult step(PointSample currentPoint) {
    LineSearchCursor direction = orientation.orient(subject, currentPoint, monitor);
    String directionType = direction.getDirectionType();
    LineSearchStrategy lineSearchStrategy;
    if (lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = lineSearchStrategyMap.get(directionType);
    }
    else {
      System.out.println(String.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = lineSearchFactory.apply(direction.getDirectionType());
      lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    PointSample previous = currentPoint;
    FailsafeLineSearchCursor wrapped = new FailsafeLineSearchCursor(direction);
    lineSearchStrategy.step(wrapped, monitor);
    return new TrainingStepResult(currentPoint, direction, previous);
  }
  
  /**
   * Measure point sample.
   *
   * @return the point sample
   */
  public PointSample resetAndMeasure() {
    PointSample currentPoint;
    int retries = 0;
    do {
      if (retries == 0 && !subject.resetSampling()) throw new IterativeStopException();
      if (10 < retries++) throw new IterativeStopException();
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
  public ValidatingTrainer setTimeout(Duration timeout) {
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
  public ValidatingTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public ValidatingTrainer setTimeout(int number, TemporalUnit units) {
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
  public ValidatingTrainer setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
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
  public ValidatingTrainer setOrientation(OrientationStrategy orientation) {
    this.orientation = orientation;
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
  public ValidatingTrainer setMonitor(TrainingMonitor monitor) {
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
  public ValidatingTrainer setCurrentIteration(AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
    return this;
  }
  
  /**
   * Gets line search factory.
   *
   * @return the line search factory
   */
  public Function<String, LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }
  
  /**
   * Sets line search factory.
   *
   * @param lineSearchFactory the line search factory
   * @return the line search factory
   */
  public ValidatingTrainer setLineSearchFactory(Function<String, LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
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
  public ValidatingTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
  private class TrainingStepResult {
    private PointSample currentPoint;
    private LineSearchCursor direction;
    private PointSample previous;
  
    public TrainingStepResult(PointSample currentPoint, LineSearchCursor direction, PointSample previous) {
      this.currentPoint = currentPoint;
      this.direction = direction;
      this.previous = previous;
    }
  
    public PointSample getCurrentPoint() {
      return currentPoint;
    }
    
    public LineSearchCursor getDirection() {
      return direction;
    }
    
    public PointSample getPrevious() {
      return previous;
    }
    
    public TrainingStepResult invoke() {
      return this;
    }
  }
}
