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

import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.FailsafeLineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.QQN;
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
  
  private final StochasticArrayTrainable trainingSubject;
  private final Trainable validationSubject;
  private Duration timeout;
  private double terminateThreshold;
  private OrientationStrategy orientation = new QQN();
  private Function<String, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
  private Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  private TrainingMonitor monitor = new TrainingMonitor();
  private int maxIterations = Integer.MAX_VALUE;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int epochIterations = 1;
  private int trainingSize = 10000;
  private double trainingTarget = 0.9;
  private double overtrainingTarget = 2;
  private int minEpochIterations = 1;
  private int maxEpochIterations = 50;
  private int minTrainingSize = 100;
  private int maxTrainingSize = 1000000;
  private double adjustmentTolerance = 0.1;
  private double adjustmentFactor = 0.5;
  private int disappointmentThreshold = 0;
  private double pessimism = 3;
  private final AtomicInteger disappointments = new AtomicInteger(0);
  
  /**
   * Instantiates a new Iterative trainer.
   *
   * @param trainingSubject   the subject
   * @param validationSubject the validation subject
   */
  public ValidatingTrainer(StochasticArrayTrainable trainingSubject, Trainable validationSubject) {
    this.trainingSubject = trainingSubject;
    this.validationSubject = validationSubject;
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
    EpochParams epochParams = new EpochParams(timeoutMs, epochIterations, getTrainingSize(), validationSubject.measure());
    PointSample currentPoint = resetAndMeasure();
    while (timeoutMs > System.currentTimeMillis() && currentPoint.getMean() > terminateThreshold) {
      if (shouldHalt(timeoutMs)) break;
      monitor.log(String.format("Epoch parameters: %s, %s", epochParams.trainingSize, epochParams.iterations));
      EpochResult epochResult = epoch(epochParams);
      double adj1 = Math.pow(Math.log(getTrainingTarget()) / Math.log(epochResult.getValidationDelta()), adjustmentFactor);
      double adj2 = Math.pow(epochResult.getOverTrainingCoeff() / getOvertrainingTarget(), adjustmentFactor);
      boolean antivalidated = Math.random() > Math.pow((1 - epochResult.getValidationDelta()), pessimism);
      boolean saturated = epochParams.trainingSize >= getMaxTrainingSize();
      monitor.log(String.format("Epoch result with %s iter, %s samples: {validation delta = %.6f; training delta = %.6f; Overtraining = %.3f}, {%.3f, %.3f}",
        epochResult.iterations, epochParams.trainingSize, epochResult.getValidationDelta(), epochResult.getTrainingDelta(), epochResult.getOverTrainingCoeff(), adj1, adj2));
      if (!epochResult.continueTraining) break;
      if (antivalidated && saturated) {
        if(disappointments.incrementAndGet() > getDisappointmentThreshold()) {
          monitor.log("Training converged");
          break;
        }
      } else {
        disappointments.set(0);
      }
      if (epochResult.getValidationDelta() < 1.0 && epochResult.getTrainingDelta() < 1.0) {
        if (adj1 < (1 - adjustmentTolerance) || adj1 > (1 + adjustmentTolerance)) {
          epochParams.iterations = Math.max(getMinEpochIterations(), Math.min(getMaxEpochIterations(), (int) (epochResult.iterations * adj1)));
        }
        if (adj2 < (1 + adjustmentTolerance) || adj2 > (1 - adjustmentTolerance)) {
          epochParams.trainingSize = Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), (int) (epochParams.trainingSize * adj2)));
        }
      }
      else {
        epochParams.trainingSize = Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), epochParams.trainingSize * 2));
      }
      epochParams.priorValidation = epochResult.currentValidation;
      orientation.reset();
    }
    monitor.log("Training terminated");
    return null == currentPoint ? Double.NaN : currentPoint.getMean();
  }
  
  /**
   * Epoch epoch result.
   *
   * @param epochParams the epoch params
   * @return the epoch result
   */
  protected EpochResult epoch(EpochParams epochParams) {
    trainingSubject.setTrainingSize(epochParams.trainingSize);
    PointSample currentPoint = resetAndMeasure();
    PointSample priorPoint = currentPoint.copyDelta();
    assert (0 < currentPoint.delta.map.size()) : "Nothing to optimize";
    int step = 1;
    for (; step <= epochParams.iterations || epochParams.iterations <= 0; step++) {
      if (shouldHalt(epochParams.timeoutMs)) {
        return new EpochResult(false, epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, step);
      }
      StepResult epoch = step(currentPoint);
      currentPoint = epoch.currentPoint.setRate(0.0);
      if (epoch.previous.getMean() <= epoch.currentPoint.getMean()) {
        return new EpochResult(reset(epoch.currentPoint.getMean()), epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, step);
      } else {
        monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), epoch.currentPoint.getMean()));
      }
      monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
    }
    return new EpochResult(true, epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, step);
  }
  
  /**
   * Step step result.
   *
   * @param previousPoint the previous point
   * @return the step result
   */
  protected StepResult step(PointSample previousPoint) {
    currentIteration.incrementAndGet();
    LineSearchCursor direction = orientation.orient(trainingSubject, previousPoint, monitor);
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
    FailsafeLineSearchCursor cursor = new FailsafeLineSearchCursor(direction, previousPoint, monitor);
    lineSearchStrategy.step(cursor, monitor);
    PointSample bestPoint = cursor.getBest(monitor).reset();
    if(bestPoint.getMean() > previousPoint.getMean()) throw new IllegalStateException(bestPoint.getMean() +" > "+previousPoint.getMean());
    return new StepResult(previousPoint, bestPoint);
  }
  
  private boolean reset(double value) {
    if (trainingSubject.resetSampling()) {
      monitor.log(String.format("Iteration %s failed, retrying. Error: %s", currentIteration.get(), value));
      return true;
    }
    else {
      monitor.log(String.format("Iteration %s failed, aborting. Error: %s", currentIteration.get(), value));
      return false;
    }
  }
  
  /**
   * Should halt boolean.
   *
   * @param timeoutMs the timeout ms
   * @return the boolean
   */
  protected boolean shouldHalt(long timeoutMs) {
    boolean stopTraining = timeoutMs < System.currentTimeMillis();
    stopTraining |= currentIteration.get() > maxIterations;
    return stopTraining;
  }
  
  
  /**
   * Reset and measure point sample.
   *
   * @return the point sample
   */
  protected PointSample resetAndMeasure() {
    //currentIteration.incrementAndGet();
    if (!trainingSubject.resetSampling()) throw new IterativeStopException();
    int retries = 0;
    do {
      if (10 < retries++) throw new IterativeStopException();
      PointSample currentPoint = trainingSubject.measure();
      if (Double.isFinite(currentPoint.getMean())) return currentPoint;
    } while (true);
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
  public int getEpochIterations() {
    return epochIterations;
  }
  
  /**
   * Sets iterations per sample.
   *
   * @param epochIterations the iterations per sample
   * @return the iterations per sample
   */
  public ValidatingTrainer setEpochIterations(int epochIterations) {
    this.epochIterations = epochIterations;
    return this;
  }
  
  /**
   * Gets validation subject.
   *
   * @return the validation subject
   */
  public Trainable getValidationSubject() {
    return validationSubject;
  }
  
  /**
   * Gets training size.
   *
   * @return the training size
   */
  public int getTrainingSize() {
    return trainingSize;
  }
  
  /**
   * Sets training size.
   *
   * @param trainingSize the training size
   * @return the training size
   */
  public ValidatingTrainer setTrainingSize(int trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }
  
  /**
   * Gets training target.
   *
   * @return the training target
   */
  public double getTrainingTarget() {
    return trainingTarget;
  }
  
  /**
   * Sets training target.
   *
   * @param trainingTarget the training target
   * @return the training target
   */
  public ValidatingTrainer setTrainingTarget(double trainingTarget) {
    this.trainingTarget = trainingTarget;
    return this;
  }
  
  /**
   * Gets overtraining target.
   *
   * @return the overtraining target
   */
  public double getOvertrainingTarget() {
    return overtrainingTarget;
  }
  
  /**
   * Sets overtraining target.
   *
   * @param overtrainingTarget the overtraining target
   * @return the overtraining target
   */
  public ValidatingTrainer setOvertrainingTarget(double overtrainingTarget) {
    this.overtrainingTarget = overtrainingTarget;
    return this;
  }
  
  /**
   * Gets min epoch iterations.
   *
   * @return the min epoch iterations
   */
  public int getMinEpochIterations() {
    return minEpochIterations;
  }
  
  /**
   * Sets min epoch iterations.
   *
   * @param minEpochIterations the min epoch iterations
   * @return the min epoch iterations
   */
  public ValidatingTrainer setMinEpochIterations(int minEpochIterations) {
    this.minEpochIterations = minEpochIterations;
    return this;
  }
  
  /**
   * Gets max epoch iterations.
   *
   * @return the max epoch iterations
   */
  public int getMaxEpochIterations() {
    return maxEpochIterations;
  }
  
  /**
   * Sets max epoch iterations.
   *
   * @param maxEpochIterations the max epoch iterations
   * @return the max epoch iterations
   */
  public ValidatingTrainer setMaxEpochIterations(int maxEpochIterations) {
    this.maxEpochIterations = maxEpochIterations;
    return this;
  }
  
  /**
   * Gets min training size.
   *
   * @return the min training size
   */
  public int getMinTrainingSize() {
    return minTrainingSize;
  }
  
  /**
   * Sets min training size.
   *
   * @param minTrainingSize the min training size
   * @return the min training size
   */
  public ValidatingTrainer setMinTrainingSize(int minTrainingSize) {
    this.minTrainingSize = minTrainingSize;
    return this;
  }
  
  /**
   * Gets max training size.
   *
   * @return the max training size
   */
  public int getMaxTrainingSize() {
    return maxTrainingSize;
  }
  
  /**
   * Sets max training size.
   *
   * @param maxTrainingSize the max training size
   * @return the max training size
   */
  public ValidatingTrainer setMaxTrainingSize(int maxTrainingSize) {
    this.maxTrainingSize = maxTrainingSize;
    return this;
  }
  
  /**
   * Gets adjustment tolerance.
   *
   * @return the adjustment tolerance
   */
  public double getAdjustmentTolerance() {
    return adjustmentTolerance;
  }
  
  /**
   * Sets adjustment tolerance.
   *
   * @param adjustmentTolerance the adjustment tolerance
   * @return the adjustment tolerance
   */
  public ValidatingTrainer setAdjustmentTolerance(double adjustmentTolerance) {
    this.adjustmentTolerance = adjustmentTolerance;
    return this;
  }
  
  /**
   * Gets adjustment factor.
   *
   * @return the adjustment factor
   */
  public double getAdjustmentFactor() {
    return adjustmentFactor;
  }
  
  /**
   * Sets adjustment factor.
   *
   * @param adjustmentFactor the adjustment factor
   * @return the adjustment factor
   */
  public ValidatingTrainer setAdjustmentFactor(double adjustmentFactor) {
    this.adjustmentFactor = adjustmentFactor;
    return this;
  }
  
  public int getDisappointmentThreshold() {
    return disappointmentThreshold;
  }
  
  public void setDisappointmentThreshold(int disappointmentThreshold) {
    this.disappointmentThreshold = disappointmentThreshold;
  }
  
  public double getPessimism() {
    return pessimism;
  }
  
  public void setPessimism(double pessimism) {
    this.pessimism = pessimism;
  }
  
  private class StepResult {
    /**
     * The Current point.
     */
    PointSample currentPoint;
    /**
     * The Previous.
     */
    PointSample previous;

    /**
     * Instantiates a new Step result.
     *
     * @param previous     the previous
     * @param currentPoint the current point
     */
    public StepResult(PointSample previous, PointSample currentPoint) {
      this.currentPoint = currentPoint;
      this.previous = previous;
    }
    
  }
  
  private static class EpochParams {
    /**
     * The Timeout ms.
     */
    long timeoutMs;
    /**
     * The Iterations.
     */
    int iterations;
    /**
     * The Training size.
     */
    int trainingSize;
    /**
     * The Prior validation.
     */
    PointSample priorValidation;
    
    private EpochParams(long timeoutMs, int iterations, int trainingSize, PointSample priorValidation) {
      this.timeoutMs = timeoutMs;
      this.iterations = iterations;
      this.trainingSize = trainingSize;
      this.priorValidation = priorValidation;
    }
    
  }
  
  private static class EpochResult {

    /**
     * The Continue training.
     */
    boolean continueTraining;
    /**
     * The Prior validation.
     */
    PointSample priorValidation;
    /**
     * The Prior point.
     */
    PointSample priorPoint;
    /**
     * The Current validation.
     */
    PointSample currentValidation;
    /**
     * The Current point.
     */
    PointSample currentPoint;
    /**
     * The Iterations.
     */
    int iterations;

    /**
     * Instantiates a new Epoch result.
     *
     * @param continueTraining  the continue training
     * @param priorValidation   the prior validation
     * @param priorPoint        the prior point
     * @param currentValidation the current validation
     * @param currentPoint      the current point
     * @param iterations        the iterations
     */
    public EpochResult(boolean continueTraining, PointSample priorValidation, PointSample priorPoint, PointSample currentValidation, PointSample currentPoint, int iterations) {
      this.priorValidation = priorValidation;
      this.priorPoint = priorPoint;
      this.currentValidation = currentValidation;
      this.currentPoint = currentPoint;
      this.continueTraining = continueTraining;
      this.iterations = iterations;
    }

    /**
     * Gets over training coeff.
     *
     * @return the over training coeff
     */
    public double getOverTrainingCoeff() {
      return (Math.log(getTrainingDelta()) / Math.log(getValidationDelta()));
    }

    /**
     * Gets validation delta.
     *
     * @return the validation delta
     */
    public double getValidationDelta() {
      return (currentValidation.getMean() / priorValidation.getMean());
    }

    /**
     * Gets training delta.
     *
     * @return the training delta
     */
    public double getTrainingDelta() {
      return (currentPoint.getMean() / priorPoint.getMean());
    }
    
  }
}

