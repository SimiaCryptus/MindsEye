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
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
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
  
  private final StochasticArrayTrainable trainingSubject;
  private final Trainable validationSubject;
  private Duration timeout;
  private double terminateThreshold;
  private OrientationStrategy orientation = new LBFGS();
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
  
  /**
   * Instantiates a new Iterative trainer.
   *
   * @param trainingSubject the subject
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
    while (timeoutMs > System.currentTimeMillis() && currentPoint.value > terminateThreshold) {
      if (shouldHalt(timeoutMs)) break;
      monitor.log(String.format("Epoch parameters: %s, %s", epochParams.trainingSize, epochParams.iterations));
      EpochResult epochResult = epoch(epochParams);
//      monitor.log(String.format("Epoch result validation delta = %.9f; %f to %f",
//        epochResult.getValidationDelta(), epochResult.priorValidation.value, epochResult.currentValidation.value));
//      monitor.log(String.format("Epoch result training delta = %.9f; %f to %f",
//        epochResult.getTrainingDelta(), epochResult.priorPoint.value, epochResult.currentPoint.value));
      monitor.log(String.format("Epoch result validation delta = %.6f; training delta = %.6f; Overtraining = %.9f ",
        epochResult.getValidationDelta(), epochResult.getTrainingDelta(), epochResult.getOverTrainingCoeff()));
  
      //        if (value1 > Math.pow(target1, 1 - adjustmentTolerance)) {
//          epochParams.iterations = Math.max(minEpochIterations, epochParams.iterations * 2);
//        }
//        else if (value1 < Math.pow(target1, 1 + adjustmentTolerance)) {
//          epochParams.iterations = Math.min(maxEpochIterations, epochParams.iterations / 2);
//        }
      double adj1 = Math.pow(Math.log(getTrainingTarget()) / Math.log(epochResult.getValidationDelta()), adjustmentFactor);
      monitor.log(String.format("Adjustment from %f to %f: %.9f ", Math.log(epochResult.getValidationDelta()), Math.log(getTrainingTarget()), adj1));
      if(epochResult.getValidationDelta() < 1.0) {
        if(adj1 < (1-adjustmentTolerance) || adj1 > (1+adjustmentTolerance)) {
          epochParams.iterations = Math.max(getMinEpochIterations(),Math.min(getMaxEpochIterations(), (int) (epochResult.iterations * adj1)));
        }
        double adj2 = Math.pow(epochResult.getOverTrainingCoeff() / getOvertrainingTarget(), adjustmentFactor);
        if (adj2 < (1 + adjustmentTolerance) || adj2 > (1 - adjustmentTolerance)) {
          epochParams.trainingSize = Math.max(getMinTrainingSize(),Math.min(getMaxTrainingSize(), (int) (epochParams.trainingSize * adj2)));
        }
      } else {
        epochParams.trainingSize = Math.max(getMinTrainingSize(),Math.min(getMaxTrainingSize(), epochParams.trainingSize * 2));
      }

      if(!epochResult.continueTraining) break;
      epochParams.priorValidation = epochResult.currentValidation;
      orientation.reset();
    }
    return null == currentPoint ? Double.NaN : currentPoint.value;
  }
  
  protected EpochResult epoch(EpochParams epochParams) {
    trainingSubject.setTrainingSize(epochParams.trainingSize);
    PointSample currentPoint = resetAndMeasure();
    PointSample priorPoint = currentPoint.copyDelta();
    assert (0 < currentPoint.delta.map.size()) : "Nothing to optimize";
    int subiteration = 0;
    for (; subiteration < epochParams.iterations || epochParams.iterations <= 0; subiteration++) {
      if (shouldHalt(epochParams.timeoutMs))
        return new EpochResult(false, epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, subiteration);
      TrainingStepResult epoch = step(currentPoint);
      currentPoint = epoch.currentPoint.setRate(0.0);
      if (epoch.previous.value <= epoch.currentPoint.value) {
        return new EpochResult(reset(epoch.currentPoint.value), epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, subiteration);
      } else {
        monitor.log(String.format("Iteration %s complete. Error: %s", currentIteration.get(), epoch.currentPoint.value));
      }
      monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
    }
    return new EpochResult(true, epochParams.priorValidation, priorPoint, validationSubject.measure(), currentPoint, subiteration);
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
  
  protected boolean shouldHalt(long timeoutMs) {
    boolean stopTraining = timeoutMs < System.currentTimeMillis();
    stopTraining |= currentIteration.get() > maxIterations;
    return stopTraining;
  }
  
  protected TrainingStepResult step(PointSample previousPoint) {
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
    FailsafeLineSearchCursor wrapped = new FailsafeLineSearchCursor(direction, previousPoint);
    lineSearchStrategy.step(wrapped, monitor);
    PointSample currentPoint = wrapped.getBest(monitor).point;
    return new TrainingStepResult(currentPoint, direction, previousPoint);
  }
  
  protected PointSample resetAndMeasure() {
    //currentIteration.incrementAndGet();
    if (!trainingSubject.resetSampling()) throw new IterativeStopException();
    int retries = 0;
    do {
      if (10 < retries++) throw new IterativeStopException();
      PointSample currentPoint = trainingSubject.measure();
      if(Double.isFinite(currentPoint.value)) return currentPoint;
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
  
  public Trainable getValidationSubject() {
    return validationSubject;
  }
  
  public int getTrainingSize() {
    return trainingSize;
  }
  
  public void setTrainingSize(int trainingSize) {
    this.trainingSize = trainingSize;
  }
  
  public double getTrainingTarget() {
    return trainingTarget;
  }
  
  public void setTrainingTarget(double trainingTarget) {
    this.trainingTarget = trainingTarget;
  }
  
  public double getOvertrainingTarget() {
    return overtrainingTarget;
  }
  
  public void setOvertrainingTarget(double overtrainingTarget) {
    this.overtrainingTarget = overtrainingTarget;
  }
  
  public int getMinEpochIterations() {
    return minEpochIterations;
  }
  
  public void setMinEpochIterations(int minEpochIterations) {
    this.minEpochIterations = minEpochIterations;
  }
  
  public int getMaxEpochIterations() {
    return maxEpochIterations;
  }
  
  public void setMaxEpochIterations(int maxEpochIterations) {
    this.maxEpochIterations = maxEpochIterations;
  }
  
  public int getMinTrainingSize() {
    return minTrainingSize;
  }
  
  public void setMinTrainingSize(int minTrainingSize) {
    this.minTrainingSize = minTrainingSize;
  }
  
  public int getMaxTrainingSize() {
    return maxTrainingSize;
  }
  
  public void setMaxTrainingSize(int maxTrainingSize) {
    this.maxTrainingSize = maxTrainingSize;
  }
  
  public double getAdjustmentTolerance() {
    return adjustmentTolerance;
  }
  
  public void setAdjustmentTolerance(double adjustmentTolerance) {
    this.adjustmentTolerance = adjustmentTolerance;
  }
  
  public double getAdjustmentFactor() {
    return adjustmentFactor;
  }
  
  public void setAdjustmentFactor(double adjustmentFactor) {
    this.adjustmentFactor = adjustmentFactor;
  }
  
  private class TrainingStepResult {
    PointSample currentPoint;
    LineSearchCursor direction;
    PointSample previous;
  
    public TrainingStepResult(PointSample currentPoint, LineSearchCursor direction, PointSample previous) {
      this.currentPoint = currentPoint;
      this.direction = direction;
      this.previous = previous;
    }
  
  }
  
  private static class EpochParams {
    long timeoutMs;
    int iterations;
    int trainingSize;
    PointSample priorValidation;
    
    private EpochParams(long timeoutMs, int iterations, int trainingSize, PointSample priorValidation) {
      this.timeoutMs = timeoutMs;
      this.iterations = iterations;
      this.trainingSize = trainingSize;
      this.priorValidation = priorValidation;
    }
  
  }

  public static class EpochResult {

    boolean continueTraining;
    PointSample priorValidation;
    PointSample priorPoint;
    PointSample currentValidation;
    PointSample currentPoint;
    int iterations;
  
    public EpochResult(boolean continueTraining, PointSample priorValidation, PointSample priorPoint, PointSample currentValidation, PointSample currentPoint, int iterations) {
      this.priorValidation = priorValidation;
      this.priorPoint = priorPoint;
      this.currentValidation = currentValidation;
      this.currentPoint = currentPoint;
      this.continueTraining = continueTraining;
      this.iterations = iterations;
    }
  
    public double getOverTrainingCoeff() {
      return (Math.log(getTrainingDelta()) / Math.log(getValidationDelta()));
    }
  
    public double getValidationDelta() {
      return (currentValidation.value / priorValidation.value);
    }
  
    public double getTrainingDelta() {
      return (currentPoint.value / priorPoint.value);
    }
  
  }
}

