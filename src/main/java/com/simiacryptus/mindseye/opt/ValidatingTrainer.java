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

import com.simiacryptus.mindseye.eval.*;
import com.simiacryptus.mindseye.eval.SampledCachedTrainable;
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.eval.Trainable.PointSample;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.opt.line.*;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.lang.TimedResult;

import java.lang.management.ManagementFactory;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Validating trainer.
 */
public class ValidatingTrainer {
  
  
  private final List<TrainingPhase> regimen;
  private TrainingMonitor monitor = new TrainingMonitor();
  private final Trainable validationSubject;
  
  private final AtomicInteger disappointments = new AtomicInteger(0);
  private final AtomicLong trainingMeasurementTime = new AtomicLong(0);
  private final AtomicLong validatingMeasurementTime = new AtomicLong(0);
  private Duration timeout;
  private double terminateThreshold;
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
  private double pessimism = 10;
  private int improvmentStaleThreshold = 3;
  
  /**
   * Instantiates a new Validating trainer.
   *
   * @param trainingSubject   the training subject
   * @param validationSubject the validation subject
   */
  public ValidatingTrainer(SampledTrainable trainingSubject, Trainable validationSubject) {
    this.regimen = new ArrayList(Arrays.asList(new TrainingPhase(new PerformanceWrapper(trainingSubject).cached())));
    this.validationSubject = new Trainable() {
      @Override
      public PointSample measure(boolean isStatic, TrainingMonitor monitor) {
        TimedResult<PointSample> time = TimedResult.time(() ->
          validationSubject.measure(isStatic, monitor)
        );
        validatingMeasurementTime.addAndGet(time.timeNanos);
        return time.result;
      }
  
      @Override
      public boolean reseed(long seed) {
        return validationSubject.reseed(seed);
      }
    }.cached();
    this.trainingSize = trainingSubject.getTrainingSize();
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
    long timeoutAt = System.currentTimeMillis() + timeout.toMillis();
    EpochParams epochParams = new EpochParams(timeoutAt, epochIterations, getTrainingSize(), validationSubject.measure(true, monitor));
    int epochNumber = 0;
    int iterationNumber = 0;
    int lastImprovement = 0;
    double lowestValidation = Double.POSITIVE_INFINITY;
    while (true) {
      if (shouldHalt(monitor, timeoutAt)) {
        monitor.log("Training halted");
        break;
      }
      monitor.log(String.format("Epoch parameters: %s, %s", epochParams.trainingSize, epochParams.iterations));
      List<TrainingPhase> regimen = getRegimen();
      long seed = System.nanoTime();
      List<EpochResult> epochResults = IntStream.range(0, regimen.size()).mapToObj(i->{
        TrainingPhase phase = getRegimen().get(i);
        return runPhase(epochParams, phase, i, seed);
      }).collect(Collectors.toList());
      EpochResult primaryPhase = epochResults.get(0);
      iterationNumber += primaryPhase.iterations;
      double trainingDelta = (primaryPhase.currentPoint.getMean() / primaryPhase.priorPoint.getMean());
      PointSample currentValidation = validationSubject.measure(true, monitor);
      double overtraining = (Math.log((trainingDelta)) / Math.log((currentValidation.getMean() / epochParams.validation.getMean())));
      double validationDelta = (currentValidation.getMean() / epochParams.validation.getMean());
      double adj1 = Math.pow(Math.log(getTrainingTarget()) / Math.log(validationDelta), adjustmentFactor);
      double adj2 = Math.pow(overtraining / getOvertrainingTarget(), adjustmentFactor);
      double validationMean = currentValidation.getMean();
      if (validationMean < lowestValidation) {
        lowestValidation = validationMean;
        lastImprovement = iterationNumber;
      }
      monitor.log(String.format("Epoch %d result with %s iterations, %s/%s samples: {validation *= 2^%.5f; training *= 2^%.3f; Overtraining = %.2f}, {itr*=%.2f, len*=%.2f} %s since improvement; %.4f validation time",
        ++epochNumber, primaryPhase.iterations, epochParams.trainingSize, getMaxTrainingSize(),
        Math.log(validationDelta) / Math.log(2), Math.log(trainingDelta) / Math.log(2),
        overtraining, adj1, adj2, (iterationNumber - lastImprovement),
        validatingMeasurementTime.getAndSet(0) / 1e9));
      if (!primaryPhase.continueTraining) {
        monitor.log(String.format("Training %d runPhase halted", epochNumber));
        break;
      }
      if (epochParams.trainingSize >= getMaxTrainingSize()) {
        double roll = FastRandom.random();
        if (roll > Math.pow((2 - validationDelta), pessimism)) {
          monitor.log(String.format("Training randomly converged: %3f", roll));
          break;
        }
        else {
          if ((iterationNumber - lastImprovement) > improvmentStaleThreshold) {
            if (disappointments.incrementAndGet() > getDisappointmentThreshold()) {
              monitor.log(String.format("Training converged after %s iterations", (iterationNumber - lastImprovement)));
              break;
            }
            else {
              monitor.log(String.format("Training failed to converged on %s attempt after %s iterations", disappointments.get(), (iterationNumber - lastImprovement)));
            }
          }
          else {
            disappointments.set(0);
          }
        }
      }
      if (validationDelta < 1.0 && trainingDelta < 1.0) {
        if (adj1 < (1 - adjustmentTolerance) || adj1 > (1 + adjustmentTolerance)) {
          epochParams.iterations = Math.max(getMinEpochIterations(), Math.min(getMaxEpochIterations(), (int) (primaryPhase.iterations * adj1)));
        }
        if (adj2 < (1 + adjustmentTolerance) || adj2 > (1 - adjustmentTolerance)) {
          epochParams.trainingSize = Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), (int) (epochParams.trainingSize * adj2)));
        }
      } else {
        epochParams.trainingSize = Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), epochParams.trainingSize * 5));
        epochParams.iterations = 1;
      }
      epochParams.validation = currentValidation;
    }
    return epochParams.validation.getMean();
  }
  
  /**
   * Epoch runPhase result.
   *
   * @param epochParams the runPhase params
   * @param phase
   * @param i
   * @return the runPhase result
   */
  protected EpochResult runPhase(EpochParams epochParams, TrainingPhase phase, int i, long seed) {
    monitor.log(String.format("Phase %d: %s", i, phase));
    phase.trainingSubject.setTrainingSize(epochParams.trainingSize);
    monitor.log(String.format("resetAndMeasure; trainingSize=%s",epochParams.trainingSize));
    PointSample currentPoint = reset(phase, seed).measure(phase);
    PointSample priorPoint = currentPoint.copyDelta();
    assert (0 < currentPoint.delta.getMap().size()) : "Nothing to optimize";
    int step = 1;
    for (; step <= epochParams.iterations || epochParams.iterations <= 0; step++) {
      if (shouldHalt(monitor, epochParams.timeoutMs)) {
        return new EpochResult(false, priorPoint, currentPoint, step);
      }
      long startTime = System.nanoTime();
      long prevGcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum();
      StepResult epoch = runStep(currentPoint, phase);
      long newGcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum();
      long endTime = System.nanoTime();
      String performance = String.format("%s in %.3f seconds; %.3f in orientation, %.3f in gc, %.3f in line search; %.3f eval time",
        epochParams.trainingSize, (endTime - startTime) / 1e9,
        epoch.performance[0],
        (newGcTime - prevGcTime) / 1e3,
        epoch.performance[1],
        trainingMeasurementTime.getAndSet(0) / 1e9
      );
      currentPoint = epoch.currentPoint.setRate(0.0);
      if (epoch.previous.getMean() <= epoch.currentPoint.getMean()) {
        monitor.log(String.format("Iteration %s failed, aborting. Error: %s (%s)",
          currentIteration.get(), epoch.currentPoint.getMean(), performance));
        return new EpochResult(false, priorPoint, currentPoint, step);
      }
      else {
        monitor.log(String.format("Iteration %s complete. Error: %s (%s)", currentIteration.get(), epoch.currentPoint.getMean(), performance));
      }
      monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
    }
    return new EpochResult(true, priorPoint, currentPoint, step);
  }
  
  /**
   * Step runStep result.
   *
   * @param previousPoint the previous point
   * @param phase
   * @return the runStep result
   */
  protected StepResult runStep(PointSample previousPoint, TrainingPhase phase) {
    currentIteration.incrementAndGet();
    TimedResult<LineSearchCursor> timedOrientation = TimedResult.time(() -> phase.orientation.orient(phase.trainingSubject, previousPoint, monitor));
    LineSearchCursor direction = timedOrientation.result;
    String directionType = direction.getDirectionType();
    LineSearchStrategy lineSearchStrategy;
    if (phase.lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = phase.lineSearchStrategyMap.get(directionType);
    } else {
      monitor.log(String.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = phase.lineSearchFactory.apply(direction.getDirectionType());
      phase.lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    TimedResult<PointSample> timedLineSearch = TimedResult.time(() -> {
      FailsafeLineSearchCursor cursor = new FailsafeLineSearchCursor(direction, previousPoint, monitor);
      lineSearchStrategy.step(cursor, monitor);
      PointSample restore = cursor.getBest(monitor).restore();
      cursor.step(restore.rate, monitor);
      return restore;
    });
    PointSample bestPoint = timedLineSearch.result;
    if (bestPoint.getMean() > previousPoint.getMean()) {
      throw new IllegalStateException(bestPoint.getMean() + " > " + previousPoint.getMean());
    }
    monitor.log(compare(previousPoint, bestPoint));
    return new StepResult(previousPoint, bestPoint, new double[]{timedOrientation.timeNanos / 1e9, timedLineSearch.timeNanos / 1e9});
  }
  
  private String compare(PointSample previousPoint, PointSample nextPoint) {
    DeltaSet nextWeights = nextPoint.weights;
    DeltaSet prevWeights = previousPoint.weights;
    return String.format("Overall network state change: %s", prevWeights.stream()
      .collect(Collectors.groupingBy(x -> getId(x), Collectors.toList())).entrySet().stream()
      .collect(Collectors.toMap(x -> x.getKey(), list -> {
        List<Double> doubleList = list.getValue().stream().map(prevWeight -> {
          Delta dirDelta = nextWeights.getMap().get(prevWeight.layer);
          double numerator = prevWeight.deltaStatistics().rms();
          double denominator = null==dirDelta?0:dirDelta.deltaStatistics().rms();
          return (numerator / (0==denominator?1:denominator));
        }).collect(Collectors.toList());
        if(1 == doubleList.size()) return Double.toString(doubleList.get(0));
        return new DoubleStatistics().accept(doubleList.stream().mapToDouble(x->x).toArray()).toString();
      })));
  }
  
  private static String getId(Delta x) {
    String name = x.layer.getName();
    String className = x.layer.getClass().getSimpleName();
    return name.contains(className)?className:name;
  }
  
  /**
   * Should halt boolean.
   *
   * @param monitor   the monitor
   * @param timeoutMs the timeout ms
   * @return the boolean
   */
  protected boolean shouldHalt(TrainingMonitor monitor, long timeoutMs) {
    boolean stopTraining = timeoutMs < System.currentTimeMillis();
    if (timeoutMs < System.currentTimeMillis()) {
      monitor.log("Training timeout");
      return true;
    }
    else if (currentIteration.get() > maxIterations) {
      monitor.log("Training iteration overflow");
      return true;
    }
    else {
      return false;
    }
  }
  
  
  private ValidatingTrainer reset(TrainingPhase phase, long seed) {
    if (!phase.trainingSubject.reseed(seed)) throw new IterativeStopException();
    phase.orientation.reset();
    phase.trainingSubject.reseed(seed);
    return this;
  }
  
  private PointSample measure(TrainingPhase phase) {
    int retries = 0;
    do {
      if (10 < retries++) throw new IterativeStopException();
      PointSample currentPoint = phase.trainingSubject.measure(false, monitor);
      phase.orientation.reset();
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
   * Sets orientation.
   *
   * @param orientation the orientation
   * @return the orientation
   */
  @Deprecated
  public ValidatingTrainer setOrientation(OrientationStrategy orientation) {
    getRegimen().get(0).setOrientation(orientation);
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
   * Sets line search factory.
   *
   * @param lineSearchFactory the line search factory
   * @return the line search factory
   */
  @Deprecated
  public ValidatingTrainer setLineSearchFactory(Function<String, LineSearchStrategy> lineSearchFactory) {
    getRegimen().get(0).setLineSearchFactory(lineSearchFactory);
    return this;
  }
  
  /**
   * Gets runPhase iterations.
   *
   * @return the runPhase iterations
   */
  public int getEpochIterations() {
    return epochIterations;
  }
  
  /**
   * Sets runPhase iterations.
   *
   * @param epochIterations the runPhase iterations
   * @return the runPhase iterations
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
   * Gets min runPhase iterations.
   *
   * @return the min runPhase iterations
   */
  public int getMinEpochIterations() {
    return minEpochIterations;
  }
  
  /**
   * Sets min runPhase iterations.
   *
   * @param minEpochIterations the min runPhase iterations
   * @return the min runPhase iterations
   */
  public ValidatingTrainer setMinEpochIterations(int minEpochIterations) {
    this.minEpochIterations = minEpochIterations;
    return this;
  }
  
  /**
   * Gets max runPhase iterations.
   *
   * @return the max runPhase iterations
   */
  public int getMaxEpochIterations() {
    return maxEpochIterations;
  }
  
  /**
   * Sets max runPhase iterations.
   *
   * @param maxEpochIterations the max runPhase iterations
   * @return the max runPhase iterations
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
  
  /**
   * Gets disappointment threshold.
   *
   * @return the disappointment threshold
   */
  public int getDisappointmentThreshold() {
    return disappointmentThreshold;
  }
  
  /**
   * Sets disappointment threshold.
   *
   * @param disappointmentThreshold the disappointment threshold
   */
  public void setDisappointmentThreshold(int disappointmentThreshold) {
    this.disappointmentThreshold = disappointmentThreshold;
  }
  
  /**
   * Gets pessimism.
   *
   * @return the pessimism
   */
  public double getPessimism() {
    return pessimism;
  }
  
  /**
   * Sets pessimism.
   *
   * @param pessimism the pessimism
   * @return the pessimism
   */
  public ValidatingTrainer setPessimism(double pessimism) {
    this.pessimism = pessimism;
    return this;
  }
  
  /**
   * Gets improvment stale threshold.
   *
   * @return the improvment stale threshold
   */
  public int getImprovmentStaleThreshold() {
    return improvmentStaleThreshold;
  }
  
  /**
   * Sets improvment stale threshold.
   *
   * @param improvmentStaleThreshold the improvment stale threshold
   */
  public void setImprovmentStaleThreshold(int improvmentStaleThreshold) {
    this.improvmentStaleThreshold = improvmentStaleThreshold;
  }
  
  public List<TrainingPhase> getRegimen() {
    return regimen;
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
     * The Validation.
     */
    PointSample validation;
    
    private EpochParams(long timeoutMs, int iterations, int trainingSize, PointSample validation) {
      this.timeoutMs = timeoutMs;
      this.iterations = iterations;
      this.trainingSize = trainingSize;
      this.validation = validation;
    }
    
  }
  
  private static class EpochResult {
  
    /**
     * The Continue training.
     */
    boolean continueTraining;
    /**
     * The Prior point.
     */
    PointSample priorPoint;
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
     * @param continueTraining  the continue training
     * @param priorPoint        the prior point
     * @param currentPoint      the current point
     * @param iterations        the iterations
     */
    public EpochResult(boolean continueTraining, PointSample priorPoint, PointSample currentPoint, int iterations) {
      this.priorPoint = priorPoint;
      this.currentPoint = currentPoint;
      this.continueTraining = continueTraining;
      this.iterations = iterations;
    }
  
  }
  
  private class StepResult {
    final double[] performance;
    /**
     * The Current point.
     */
    final PointSample currentPoint;
    /**
     * The Previous.
     */
    final PointSample previous;
  
    /**
     * Instantiates a new Step result.
     *  @param previous     the previous
     * @param currentPoint the current point
     * @param performance
     */
    public StepResult(PointSample previous, PointSample currentPoint, double[] performance) {
      this.currentPoint = currentPoint;
      this.previous = previous;
      this.performance = performance;
    }
    
  }
  
  
  public static class TrainingPhase {
    private SampledTrainable trainingSubject;
    private OrientationStrategy orientation = new QQN();
    private Function<String, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
    private Map<String, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
  
    public TrainingPhase(SampledTrainable trainingSubject) {
      this.setTrainingSubject(trainingSubject);
    }
  
  
    public SampledTrainable getTrainingSubject() {
      return trainingSubject;
    }
  
    public TrainingPhase setTrainingSubject(SampledTrainable trainingSubject) {
      this.trainingSubject = trainingSubject;
      return this;
    }
  
    public OrientationStrategy getOrientation() {
      return orientation;
    }
  
    public TrainingPhase setOrientation(OrientationStrategy orientation) {
      this.orientation = orientation;
      return this;
    }
  
    public Function<String, LineSearchStrategy> getLineSearchFactory() {
      return lineSearchFactory;
    }
  
    public TrainingPhase setLineSearchFactory(Function<String, LineSearchStrategy> lineSearchFactory) {
      this.lineSearchFactory = lineSearchFactory;
      return this;
    }
  
    public Map<String, LineSearchStrategy> getLineSearchStrategyMap() {
      return lineSearchStrategyMap;
    }
  
    public TrainingPhase setLineSearchStrategyMap(Map<String, LineSearchStrategy> lineSearchStrategyMap) {
      this.lineSearchStrategyMap = lineSearchStrategyMap;
      return this;
    }
  
    @Override
    public String toString() {
      return "TrainingPhase{" +
        "trainingSubject=" + trainingSubject +
        ", orientation=" + orientation +
        '}';
    }
  }
  
  
  private class PerformanceWrapper extends TrainableWrapper<SampledTrainable> implements SampledTrainable {
  
    public PerformanceWrapper(SampledTrainable trainingSubject) {
      super(trainingSubject);
    }
    
    @Override
    public int getTrainingSize() {
      return getInner().getTrainingSize();
    }
    
    @Override
    public SampledTrainable setTrainingSize(int trainingSize) {
      getInner().setTrainingSize(trainingSize);
      return this;
    }
  
    @Override
    public SampledCachedTrainable<? extends SampledTrainable> cached() {
      return new SampledCachedTrainable<>(this);
    }
  
    @Override
    public PointSample measure(boolean isStatic, TrainingMonitor monitor) {
      TimedResult<PointSample> time = TimedResult.time(() ->
        getInner().measure(isStatic, monitor)
      );
      trainingMeasurementTime.addAndGet(time.timeNanos);
      return time.result;
    }
    
  }
}

