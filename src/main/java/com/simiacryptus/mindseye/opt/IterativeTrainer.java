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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  private static final Logger log = LoggerFactory.getLogger(IterativeTrainer.class);
  
  private final Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private final Trainable subject;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int iterationsPerSample = 1;
  private Function<String, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
  private int maxIterations = Integer.MAX_VALUE;
  private TrainingMonitor monitor = new TrainingMonitor();
  private OrientationStrategy<?> orientation = new LBFGS();
  private double terminateThreshold;
  private Duration timeout;
  
  /**
   * Instantiates a new Iterative trainer.
   *
   * @param subject the subject
   */
  public IterativeTrainer(final Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
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
  public IterativeTrainer setCurrentIteration(final AtomicInteger currentIteration) {
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
  public IterativeTrainer setIterationsPerSample(final int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
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
  public IterativeTrainer setLineSearchFactory(final Function<String, LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
    return this;
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
  public IterativeTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
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
  public IterativeTrainer setMonitor(final TrainingMonitor monitor) {
    this.monitor = monitor;
    return this;
  }
  
  /**
   * Gets orientation.
   *
   * @return the orientation
   */
  public OrientationStrategy<?> getOrientation() {
    return orientation;
  }
  
  /**
   * Sets orientation.
   *
   * @param orientation the orientation
   * @return the orientation
   */
  public IterativeTrainer setOrientation(final OrientationStrategy<?> orientation) {
    this.orientation = orientation;
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
  public IterativeTrainer setTerminateThreshold(final double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
    return this;
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
  public IterativeTrainer setTimeout(final Duration timeout) {
    this.timeout = timeout;
    return this;
  }
  
  /**
   * Measure point sample.
   *
   * @param reset the reset
   * @return the point sample
   */
  public PointSample measure(boolean reset) {
    PointSample currentPoint;
    int retries = 0;
    do {
      if (reset) {
        orientation.reset();
        if (!subject.reseed(System.nanoTime())) {
          if (retries > 0) throw new IterativeStopException("Failed to reset training subject");
        }
        else {
          monitor.log(String.format("Reset training subject"));
        }
      }
      currentPoint = subject.measure(monitor);
    } while (!Double.isFinite(currentPoint.getMean()) && 10 < retries++);
    if (!Double.isFinite(currentPoint.getMean())) throw new IterativeStopException();
    return currentPoint;
  }
  
  /**
   * Run double.
   *
   * @return the double
   */
  public double run() {
    final long timeoutMs = System.currentTimeMillis() + timeout.toMillis();
    long lastIterationTime = System.nanoTime();
    PointSample currentPoint = measure(true);
    mainLoop:
    while (timeoutMs > System.currentTimeMillis() && currentPoint.getMean() > terminateThreshold) {
      if (currentIteration.get() > maxIterations) {
        break;
      }
      currentPoint = measure(true);
      assert 0 < currentPoint.delta.getMap().size() : "Nothing to optimize";
      subiterationLoop:
      for (int subiteration = 0; subiteration < iterationsPerSample || iterationsPerSample <= 0; subiteration++) {
        if (timeoutMs < System.currentTimeMillis()) {
          break mainLoop;
        }
        if (currentIteration.incrementAndGet() > maxIterations) {
          break mainLoop;
        }
        currentPoint = measure(true);
        final PointSample _currentPoint = currentPoint;
        final TimedResult<LineSearchCursor> timedOrientation = TimedResult.time(() -> orientation.orient(subject, _currentPoint, monitor));
        final LineSearchCursor direction = timedOrientation.result;
        final String directionType = direction.getDirectionType();
        final PointSample previous = currentPoint;
        final TimedResult<PointSample> timedLineSearch = TimedResult.time(() -> step(direction, directionType, previous));
        currentPoint = timedLineSearch.result;
        final long now = System.nanoTime();
        final String perfString = String.format("Total: %.4f; Orientation: %.4f; Line Search: %.4f",
          now - lastIterationTime / 1e9, timedOrientation.timeNanos / 1e9, timedLineSearch.timeNanos / 1e9);
        lastIterationTime = now;
        if (previous.getMean() <= currentPoint.getMean()) {
          if (previous.getMean() < currentPoint.getMean()) {
            monitor.log(String.format("Resetting Iteration %s", perfString));
            currentPoint = direction.step(0, monitor).point;
          }
          else {
            monitor.log(String.format("Static Iteration %s", perfString));
          }
          if (subject.reseed(System.nanoTime())) {
            monitor.log(String.format("Iteration %s failed, retrying. Error: %s",
              currentIteration.get(), currentPoint.getMean()));
            monitor.log(String.format("Previous Error: %s -> %s",
              previous.getRate(), previous.getMean()));
            break subiterationLoop;
          }
          else {
            monitor.log(String.format("Iteration %s failed, aborting. Error: %s",
              currentIteration.get(), currentPoint.getMean()));
            monitor.log(String.format("Previous Error: %s -> %s",
              previous.getRate(), previous.getMean()));
            break mainLoop;
          }
        }
        else {
          monitor.log(String.format("Iteration %s complete. Error: %s " + perfString,
            currentIteration.get(), currentPoint.getMean()));
        }
        monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
      }
    }
    return null == currentPoint ? Double.NaN : currentPoint.getMean();
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public IterativeTrainer setTimeout(final int number, final TemporalUnit units) {
    timeout = Duration.of(number, units);
    return this;
  }
  
  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  public IterativeTrainer setTimeout(final int number, final TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }
  
  /**
   * Step point sample.
   *
   * @param direction     the direction
   * @param directionType the direction type
   * @param previous      the previous
   * @return the point sample
   */
  public PointSample step(final LineSearchCursor direction, final String directionType, final PointSample previous) {
    PointSample currentPoint;
    LineSearchStrategy lineSearchStrategy;
    if (lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = lineSearchStrategyMap.get(directionType);
    }
    else {
      log.info(String.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = lineSearchFactory.apply(direction.getDirectionType());
      lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    final FailsafeLineSearchCursor wrapped = new FailsafeLineSearchCursor(direction, previous, monitor);
    lineSearchStrategy.step(wrapped, monitor);
    currentPoint = wrapped.getBest(monitor);
    return currentPoint;
  }
  
}
