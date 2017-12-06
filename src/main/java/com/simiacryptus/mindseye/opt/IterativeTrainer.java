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
import com.simiacryptus.mindseye.opt.line.FailsafeLineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.lang.TimedResult;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * The basic type of training loop, which integrates a Trainable object
 * with an Orientation and Line Search strategy
 */
public class IterativeTrainer {
  
  private final Trainable subject;
  private final Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private Duration timeout;
  private double terminateThreshold;
  private OrientationStrategy orientation = new LBFGS();
  private Function<String, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  
  /**
   * Instantiates a new Iterative trainer.
   *
   * @param subject the subject
   */
  public IterativeTrainer(Trainable subject) {
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
  public IterativeTrainer setMaxIterations(int maxIterations) {
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
    long lastIterationTime = System.nanoTime();
    PointSample currentPoint = measure();
    mainLoop:
    while (timeoutMs > System.currentTimeMillis() && currentPoint.getMean() > terminateThreshold) {
      if (currentIteration.get() > maxIterations) break;
      currentPoint = measure();
      assert (0 < currentPoint.delta.getMap().size()) : "Nothing to optimize";
      subiterationLoop:
      for (int subiteration = 0; subiteration < iterationsPerSample || iterationsPerSample <= 0; subiteration++) {
        if (timeoutMs < System.currentTimeMillis()) break mainLoop;
        if (currentIteration.incrementAndGet() > maxIterations) break mainLoop;
        PointSample _currentPoint = currentPoint;
        TimedResult<LineSearchCursor> timedOrientation = TimedResult.time(() -> orientation.orient(subject, _currentPoint, monitor));
        LineSearchCursor direction = timedOrientation.result;
        String directionType = direction.getDirectionType();
        PointSample previous = currentPoint;
        TimedResult<PointSample> timedLineSearch = TimedResult.time(() -> step(direction, directionType, previous));
        currentPoint = timedLineSearch.result;
        long now = System.nanoTime();
        String perfString = String.format("Total: %.4f; Orientation: %.4f; Line Search: %.4f",
          now - lastIterationTime / 1e9, timedOrientation.timeNanos / 1e9, timedLineSearch.timeNanos / 1e9);
        lastIterationTime = now;
        if (previous.getMean() <= currentPoint.getMean()) {
          if (previous.getMean() < currentPoint.getMean()) {
            monitor.log(String.format("Resetting Iteration " + perfString));
            currentPoint = direction.step(0, monitor).point;
          }
          if (subject.reseed(System.nanoTime())) {
            monitor.log(String.format("Iteration %s failed, retrying. Error: %s " + perfString, currentIteration.get(), currentPoint.getMean()));
            break subiterationLoop;
          }
          else {
            monitor.log(String.format("Iteration %s failed, aborting. Error: %s " + perfString, currentIteration.get(), currentPoint.getMean()));
            break mainLoop;
          }
        }
        else {
          monitor.log(String.format("Iteration %s complete. Error: %s " + perfString, currentIteration.get(), currentPoint.getMean()));
        }
        monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
      }
      orientation.reset();
    }
    return null == currentPoint ? Double.NaN : currentPoint.getMean();
  }
  
  /**
   * Step point sample.
   *
   * @param direction     the direction
   * @param directionType the direction type
   * @param previous      the previous
   * @return the point sample
   */
  public PointSample step(LineSearchCursor direction, String directionType, PointSample previous) {
    PointSample currentPoint;
    LineSearchStrategy lineSearchStrategy;
    if (lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = lineSearchStrategyMap.get(directionType);
    }
    else {
      System.out.println(String.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = lineSearchFactory.apply(direction.getDirectionType());
      lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    FailsafeLineSearchCursor wrapped = new FailsafeLineSearchCursor(direction, previous, monitor);
    lineSearchStrategy.step(wrapped, monitor);
    currentPoint = wrapped.getBest(monitor);
    return currentPoint;
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
    } while (!Double.isFinite(currentPoint.getMean()));
    assert (Double.isFinite(currentPoint.getMean()));
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
  public IterativeTrainer setTimeout(Duration timeout) {
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
  public IterativeTrainer setTimeout(int number, TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public IterativeTrainer setTimeout(int number, TemporalUnit units) {
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
  public IterativeTrainer setTerminateThreshold(double terminateThreshold) {
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
  public IterativeTrainer setOrientation(OrientationStrategy orientation) {
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
  public IterativeTrainer setMonitor(TrainingMonitor monitor) {
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
  public IterativeTrainer setCurrentIteration(AtomicInteger currentIteration) {
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
  public IterativeTrainer setLineSearchFactory(Function<String, LineSearchStrategy> lineSearchFactory) {
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
  public IterativeTrainer setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
    return this;
  }
  
}
