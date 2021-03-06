/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.mindseye.eval.*;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.StochasticComponent;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.FailsafeLineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.LineSearchStrategy;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.DoubleStatistics;

import javax.annotation.Nonnull;
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
 * This training class attempts to manage the sample size hyperparameter and convergence criteria via the periodic
 * evaluation of a validation function. It uses an itermediate "epoch" loop where a target learning improvement and
 * overtraining ratio are sought. It also supports multiple stages to each epoch, allowing use cases such as GAN and
 * layerwise training.
 */
public class ValidatingTrainer {


  private final AtomicInteger disappointments = new AtomicInteger(0);
  @Nonnull
  private final List<TrainingPhase> regimen;
  private final AtomicLong trainingMeasurementTime = new AtomicLong(0);
  private final AtomicLong validatingMeasurementTime = new AtomicLong(0);
  @Nonnull
  private final Trainable validationSubject;
  private double adjustmentFactor = 0.5;
  private double adjustmentTolerance = 0.1;
  private AtomicInteger currentIteration = new AtomicInteger(0);
  private int disappointmentThreshold = 0;
  private int epochIterations = 1;
  private int improvmentStaleThreshold = 3;
  private int maxEpochIterations = 20;
  private int maxIterations = Integer.MAX_VALUE;
  private int maxTrainingSize = Integer.MAX_VALUE;
  private int minEpochIterations = 1;
  private int minTrainingSize = 100;
  private TrainingMonitor monitor = new TrainingMonitor();
  private double overtrainingTarget = 2;
  private double pessimism = 10;
  private double terminateThreshold;
  private Duration timeout;
  private int trainingSize = 10000;
  private double trainingTarget = 0.7;

  /**
   * Instantiates a new Validating trainer.
   *
   * @param trainingSubject   the training subject
   * @param validationSubject the validation subject
   */
  public ValidatingTrainer(@Nonnull final SampledTrainable trainingSubject, @Nonnull final Trainable validationSubject) {
    regimen = new ArrayList<TrainingPhase>(Arrays.asList(new TrainingPhase(new PerformanceWrapper(trainingSubject))));
    validationSubject.addRef();
    this.validationSubject = new TrainableBase() {
      @Override
      protected void _free() {
        validationSubject.freeRef();
      }

      @Override
      public PointSample measure(final TrainingMonitor monitor) {
        @Nonnull final TimedResult<PointSample> time = TimedResult.time(() ->
            validationSubject.measure(monitor)
        );
        validatingMeasurementTime.addAndGet(time.timeNanos);
        return time.result;
      }

      @Override
      public boolean reseed(final long seed) {
        return validationSubject.reseed(seed);
      }

      @Override
      public Layer getLayer() {
        return validationSubject.getLayer();
      }
    };
    trainingSize = trainingSubject.getTrainingSize();
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = Double.NEGATIVE_INFINITY;
  }

  @Nonnull
  private static CharSequence getId(@Nonnull final DoubleBuffer<UUID> x) {
    return x.key.toString();
  }

  private String compare(@Nonnull final PointSample previousPoint, @Nonnull final PointSample nextPoint) {
    @Nonnull final StateSet<UUID> nextWeights = nextPoint.weights;
    @Nonnull final StateSet<UUID> prevWeights = previousPoint.weights;
    return String.format("Overall network state change: %s", prevWeights.stream()
        .collect(Collectors.groupingBy(x -> x, Collectors.toList())).entrySet().stream()
        .collect(Collectors.toMap(x -> x.getKey(), list -> {
          final List<Double> doubleList = list.getValue().stream().map(prevWeight -> {
            final DoubleBuffer<UUID> dirDelta = nextWeights.getMap().get(prevWeight.key);
            final double numerator = prevWeight.deltaStatistics().rms();
            final double denominator = null == dirDelta ? 0 : dirDelta.deltaStatistics().rms();
            return numerator / (0 == denominator ? 1 : denominator);
          }).collect(Collectors.toList());
          if (1 == doubleList.size())
            return Double.toString(doubleList.get(0));
          return new DoubleStatistics().accept(doubleList.stream().mapToDouble(x -> x).toArray()).toString();
        })));
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
  @Nonnull
  public ValidatingTrainer setAdjustmentFactor(final double adjustmentFactor) {
    this.adjustmentFactor = adjustmentFactor;
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
  @Nonnull
  public ValidatingTrainer setAdjustmentTolerance(final double adjustmentTolerance) {
    this.adjustmentTolerance = adjustmentTolerance;
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
  @Nonnull
  public ValidatingTrainer setCurrentIteration(final AtomicInteger currentIteration) {
    this.currentIteration = currentIteration;
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
  public void setDisappointmentThreshold(final int disappointmentThreshold) {
    this.disappointmentThreshold = disappointmentThreshold;
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
  @Nonnull
  public ValidatingTrainer setEpochIterations(final int epochIterations) {
    this.epochIterations = epochIterations;
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
  public void setImprovmentStaleThreshold(final int improvmentStaleThreshold) {
    this.improvmentStaleThreshold = improvmentStaleThreshold;
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
  @Nonnull
  public ValidatingTrainer setMaxEpochIterations(final int maxEpochIterations) {
    this.maxEpochIterations = maxEpochIterations;
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
  @Nonnull
  public ValidatingTrainer setMaxIterations(final int maxIterations) {
    this.maxIterations = maxIterations;
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
  @Nonnull
  public ValidatingTrainer setMaxTrainingSize(final int maxTrainingSize) {
    this.maxTrainingSize = maxTrainingSize;
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
  @Nonnull
  public ValidatingTrainer setMinEpochIterations(final int minEpochIterations) {
    this.minEpochIterations = minEpochIterations;
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
  @Nonnull
  public ValidatingTrainer setMinTrainingSize(final int minTrainingSize) {
    this.minTrainingSize = minTrainingSize;
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
  @Nonnull
  public ValidatingTrainer setMonitor(final TrainingMonitor monitor) {
    this.monitor = monitor;
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
  @Nonnull
  public ValidatingTrainer setOvertrainingTarget(final double overtrainingTarget) {
    this.overtrainingTarget = overtrainingTarget;
    return this;
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
  @Nonnull
  public ValidatingTrainer setPessimism(final double pessimism) {
    this.pessimism = pessimism;
    return this;
  }

  /**
   * Gets regimen.
   *
   * @return the regimen
   */
  @Nonnull
  public List<TrainingPhase> getRegimen() {
    return regimen;
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
  @Nonnull
  public ValidatingTrainer setTerminateThreshold(final double terminateThreshold) {
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
  @Nonnull
  public ValidatingTrainer setTimeout(final Duration timeout) {
    this.timeout = timeout;
    return this;
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
  @Nonnull
  public ValidatingTrainer setTrainingSize(final int trainingSize) {
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
  @Nonnull
  public ValidatingTrainer setTrainingTarget(final double trainingTarget) {
    this.trainingTarget = trainingTarget;
    return this;
  }

  /**
   * Gets validation subject.
   *
   * @return the validation subject
   */
  @Nonnull
  public Trainable getValidationSubject() {
    return validationSubject;
  }

  private PointSample measure(@Nonnull final TrainingPhase phase) {
    int retries = 0;
    do {
      if (10 < retries++) throw new IterativeStopException();
      final PointSample currentPoint = phase.trainingSubject.measure(monitor);
      if (Double.isFinite(currentPoint.getMean())) return currentPoint;
      phase.orientation.reset();
    } while (true);
  }

  @Nonnull
  private ValidatingTrainer reset(@Nonnull final TrainingPhase phase, final long seed) {
    if (!phase.trainingSubject.reseed(seed)) throw new IterativeStopException();
    phase.orientation.reset();
    phase.trainingSubject.reseed(seed);
    if (phase.trainingSubject.getLayer() instanceof DAGNetwork) {
      ((DAGNetwork) phase.trainingSubject.getLayer()).visitLayers(layer -> {
        if (layer instanceof StochasticComponent)
          ((StochasticComponent) layer).shuffle(StochasticComponent.random.get().nextLong());
      });
    }
    return this;
  }

  /**
   * Run double.
   *
   * @return the double
   */
  public double run() {
    try {
      final long timeoutAt = System.currentTimeMillis() + timeout.toMillis();
      if (validationSubject.getLayer() instanceof DAGNetwork) {
        ((DAGNetwork) validationSubject.getLayer()).visitLayers(layer -> {
          if (layer instanceof StochasticComponent) ((StochasticComponent) layer).clearNoise();
        });
      }
      @Nonnull final EpochParams epochParams = new EpochParams(timeoutAt, epochIterations, getTrainingSize(), validationSubject.measure(monitor));
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
        @Nonnull final List<TrainingPhase> regimen = getRegimen();
        final long seed = System.nanoTime();
        final List<EpochResult> epochResults = IntStream.range(0, regimen.size()).mapToObj(i -> {
          final TrainingPhase phase = getRegimen().get(i);
          return runPhase(epochParams, phase, i, seed);
        }).collect(Collectors.toList());
        final EpochResult primaryPhase = epochResults.get(0);
        iterationNumber += primaryPhase.iterations;
        final double trainingDelta = primaryPhase.currentPoint.getMean() / primaryPhase.priorMean;
        if (validationSubject.getLayer() instanceof DAGNetwork) {
          ((DAGNetwork) validationSubject.getLayer()).visitLayers(layer -> {
            if (layer instanceof StochasticComponent) ((StochasticComponent) layer).clearNoise();
          });
        }
        final PointSample currentValidation = validationSubject.measure(monitor);
        final double overtraining = Math.log(trainingDelta) / Math.log(currentValidation.getMean() / epochParams.validation.getMean());
        final double validationDelta = currentValidation.getMean() / epochParams.validation.getMean();
        final double adj1 = Math.pow(Math.log(getTrainingTarget()) / Math.log(validationDelta), adjustmentFactor);
        final double adj2 = Math.pow(overtraining / getOvertrainingTarget(), adjustmentFactor);
        final double validationMean = currentValidation.getMean();
        if (validationMean < lowestValidation) {
          lowestValidation = validationMean;
          lastImprovement = iterationNumber;
        }
        monitor.log(String.format("Epoch %d result apply %s iterations, %s/%s samples: {validation *= 2^%.5f; training *= 2^%.3f; Overtraining = %.2f}, {itr*=%.2f, len*=%.2f} %s since improvement; %.4f validation time",
            ++epochNumber, primaryPhase.iterations, epochParams.trainingSize, getMaxTrainingSize(),
            Math.log(validationDelta) / Math.log(2), Math.log(trainingDelta) / Math.log(2),
            overtraining, adj1, adj2, iterationNumber - lastImprovement,
            validatingMeasurementTime.getAndSet(0) / 1e9));
        if (!primaryPhase.continueTraining) {
          monitor.log(String.format("Training %d runPhase halted", epochNumber));
          break;
        }
        if (epochParams.trainingSize >= getMaxTrainingSize()) {
          final double roll = FastRandom.INSTANCE.random();
          if (roll > Math.pow(2 - validationDelta, pessimism)) {
            monitor.log(String.format("Training randomly converged: %3f", roll));
            break;
          } else {
            if (iterationNumber - lastImprovement > improvmentStaleThreshold) {
              if (disappointments.incrementAndGet() > getDisappointmentThreshold()) {
                monitor.log(String.format("Training converged after %s iterations", iterationNumber - lastImprovement));
                break;
              } else {
                monitor.log(String.format("Training failed to converged on %s attempt after %s iterations", disappointments.get(), iterationNumber - lastImprovement));
              }
            } else {
              disappointments.set(0);
            }
          }
        }
        if (validationDelta < 1.0 && trainingDelta < 1.0) {
          if (adj1 < 1 - adjustmentTolerance || adj1 > 1 + adjustmentTolerance) {
            epochParams.iterations = Math.max(getMinEpochIterations(), Math.min(getMaxEpochIterations(), (int) (primaryPhase.iterations * adj1)));
          }
          if (adj2 < 1 + adjustmentTolerance || adj2 > 1 - adjustmentTolerance) {
            epochParams.trainingSize = Math.max(0, Math.min(Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), (int) (epochParams.trainingSize * adj2))), epochParams.trainingSize));
          }
        } else {
          epochParams.trainingSize = Math.max(0, Math.min(Math.max(getMinTrainingSize(), Math.min(getMaxTrainingSize(), epochParams.trainingSize * 5)), epochParams.trainingSize));
          epochParams.iterations = 1;
        }
        epochParams.validation = currentValidation;
      }
      if (validationSubject.getLayer() instanceof DAGNetwork) {
        ((DAGNetwork) validationSubject.getLayer()).visitLayers(layer -> {
          if (layer instanceof StochasticComponent) ((StochasticComponent) layer).clearNoise();
        });
      }
      return epochParams.validation.getMean();
    } catch (@Nonnull final Throwable e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Epoch runPhase result.
   *
   * @param epochParams the runPhase params
   * @param phase       the phase
   * @param i           the
   * @param seed        the seed
   * @return the runPhase result
   */
  @Nonnull
  protected EpochResult runPhase(@Nonnull final EpochParams epochParams, @Nonnull final TrainingPhase phase, final int i, final long seed) {
    monitor.log(String.format("Phase %d: %s", i, phase));
    phase.trainingSubject.setTrainingSize(epochParams.trainingSize);
    monitor.log(String.format("resetAndMeasure; trainingSize=%s", epochParams.trainingSize));
    PointSample currentPoint = reset(phase, seed).measure(phase);
    final double pointMean = currentPoint.getMean();
    assert 0 < currentPoint.delta.getMap().size() : "Nothing to optimize";
    int step = 1;
    for (; step <= epochParams.iterations || epochParams.iterations <= 0; step++) {
      if (shouldHalt(monitor, epochParams.timeoutMs)) {
        return new EpochResult(false, pointMean, currentPoint, step);
      }
      final long startTime = System.nanoTime();
      final long prevGcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum();
      @Nonnull final StepResult epoch = runStep(currentPoint, phase);
      final long newGcTime = ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(x -> x.getCollectionTime()).sum();
      final long endTime = System.nanoTime();
      final CharSequence performance = String.format("%s in %.3f seconds; %.3f in orientation, %.3f in gc, %.3f in line search; %.3f trainAll time",
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
        return new EpochResult(false, pointMean, currentPoint, step);
      } else {
        monitor.log(String.format("Iteration %s complete. Error: %s (%s)", currentIteration.get(), epoch.currentPoint.getMean(), performance));
      }
      monitor.onStepComplete(new Step(currentPoint, currentIteration.get()));
    }
    return new EpochResult(true, pointMean, currentPoint, step);
  }

  /**
   * Step runStep result.
   *
   * @param previousPoint the previous point
   * @param phase         the phase
   * @return the runStep result
   */
  @Nonnull
  protected StepResult runStep(@Nonnull final PointSample previousPoint, @Nonnull final TrainingPhase phase) {
    currentIteration.incrementAndGet();
    @Nonnull final TimedResult<LineSearchCursor> timedOrientation = TimedResult.time(() -> phase.orientation.orient(phase.trainingSubject, previousPoint, monitor));
    final LineSearchCursor direction = timedOrientation.result;
    final CharSequence directionType = direction.getDirectionType();
    LineSearchStrategy lineSearchStrategy;
    if (phase.lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = phase.lineSearchStrategyMap.get(directionType);
    } else {
      monitor.log(String.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = phase.lineSearchFactory.apply(direction.getDirectionType());
      phase.lineSearchStrategyMap.put(directionType, lineSearchStrategy);
    }
    @Nonnull final TimedResult<PointSample> timedLineSearch = TimedResult.time(() -> {
      @Nonnull final FailsafeLineSearchCursor cursor = new FailsafeLineSearchCursor(direction, previousPoint, monitor);
      lineSearchStrategy.step(cursor, monitor);
      @Nonnull final PointSample restore = cursor.getBest(monitor).restore();
      //cursor.step(restore.rate, monitor);
      return restore;
    });
    final PointSample bestPoint = timedLineSearch.result;
    if (bestPoint.getMean() > previousPoint.getMean()) {
      throw new IllegalStateException(bestPoint.getMean() + " > " + previousPoint.getMean());
    }
    monitor.log(compare(previousPoint, bestPoint));
    return new StepResult(previousPoint, bestPoint, new double[]{timedOrientation.timeNanos / 1e9, timedLineSearch.timeNanos / 1e9});
  }

  /**
   * Sets line search factory.
   *
   * @param lineSearchFactory the line search factory
   * @return the line search factory
   */
  @Nonnull
  @Deprecated
  public ValidatingTrainer setLineSearchFactory(final Function<CharSequence, LineSearchStrategy> lineSearchFactory) {
    getRegimen().get(0).setLineSearchFactory(lineSearchFactory);
    return this;
  }

  /**
   * Sets orientation.
   *
   * @param orientation the orientation
   * @return the orientation
   */
  @Nonnull
  @Deprecated
  public ValidatingTrainer setOrientation(final OrientationStrategy<?> orientation) {
    getRegimen().get(0).setOrientation(orientation);
    return this;
  }

  /**
   * Sets timeout.
   *
   * @param number the number
   * @param units  the units
   * @return the timeout
   */
  @Nonnull
  public ValidatingTrainer setTimeout(final int number, @Nonnull final TemporalUnit units) {
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
  @Nonnull
  public ValidatingTrainer setTimeout(final int number, @Nonnull final TimeUnit units) {
    return setTimeout(number, Util.cvt(units));
  }

  /**
   * Should halt boolean.
   *
   * @param monitor   the monitor
   * @param timeoutMs the timeout ms
   * @return the boolean
   */
  protected boolean shouldHalt(@Nonnull final TrainingMonitor monitor, final long timeoutMs) {
    System.currentTimeMillis();
    if (timeoutMs < System.currentTimeMillis()) {
      monitor.log("Training timeout");
      return true;
    } else if (currentIteration.get() > maxIterations) {
      monitor.log("Training iteration overflow");
      return true;
    } else {
      return false;
    }
  }

  private static class EpochParams {
    /**
     * The Iterations.
     */
    int iterations;
    /**
     * The Timeout ms.
     */
    long timeoutMs;
    /**
     * The Training size.
     */
    int trainingSize;
    /**
     * The Validation.
     */
    PointSample validation;

    private EpochParams(final long timeoutMs, final int iterations, final int trainingSize, final PointSample validation) {
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
     * The Current point.
     */
    PointSample currentPoint;
    /**
     * The Iterations.
     */
    int iterations;
    /**
     * The Prior point.
     */
    double priorMean;

    /**
     * Instantiates a new Epoch result.
     *
     * @param continueTraining the continue training
     * @param priorMean        the prior point
     * @param currentPoint     the current point
     * @param iterations       the iterations
     */
    public EpochResult(final boolean continueTraining, final double priorMean, final PointSample currentPoint, final int iterations) {
      this.priorMean = priorMean;
      this.currentPoint = currentPoint;
      this.continueTraining = continueTraining;
      this.iterations = iterations;
    }

  }

  /**
   * The type Training phase.
   */
  public static class TrainingPhase {
    private Function<CharSequence, LineSearchStrategy> lineSearchFactory = (s) -> new ArmijoWolfeSearch();
    private Map<CharSequence, LineSearchStrategy> lineSearchStrategyMap = new HashMap<>();
    private OrientationStrategy<?> orientation = new QQN();
    private SampledTrainable trainingSubject;

    /**
     * Instantiates a new Training phase.
     *
     * @param trainingSubject the training subject
     */
    public TrainingPhase(final SampledTrainable trainingSubject) {
      setTrainingSubject(trainingSubject);
    }


    /**
     * Gets line search factory.
     *
     * @return the line search factory
     */
    public Function<CharSequence, LineSearchStrategy> getLineSearchFactory() {
      return lineSearchFactory;
    }

    /**
     * Sets line search factory.
     *
     * @param lineSearchFactory the line search factory
     * @return the line search factory
     */
    @Nonnull
    public TrainingPhase setLineSearchFactory(final Function<CharSequence, LineSearchStrategy> lineSearchFactory) {
      this.lineSearchFactory = lineSearchFactory;
      return this;
    }

    /**
     * Gets line search strategy buildMap.
     *
     * @return the line search strategy buildMap
     */
    public Map<CharSequence, LineSearchStrategy> getLineSearchStrategyMap() {
      return lineSearchStrategyMap;
    }

    /**
     * Sets line search strategy buildMap.
     *
     * @param lineSearchStrategyMap the line search strategy buildMap
     * @return the line search strategy buildMap
     */
    @Nonnull
    public TrainingPhase setLineSearchStrategyMap(final Map<CharSequence, LineSearchStrategy> lineSearchStrategyMap) {
      this.lineSearchStrategyMap = lineSearchStrategyMap;
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
    @Nonnull
    public TrainingPhase setOrientation(final OrientationStrategy<?> orientation) {
      this.orientation = orientation;
      return this;
    }

    /**
     * Gets training subject.
     *
     * @return the training subject
     */
    public SampledTrainable getTrainingSubject() {
      return trainingSubject;
    }

    /**
     * Sets training subject.
     *
     * @param trainingSubject the training subject
     * @return the training subject
     */
    @Nonnull
    public TrainingPhase setTrainingSubject(final SampledTrainable trainingSubject) {
      this.trainingSubject = trainingSubject;
      return this;
    }

    @Nonnull
    @Override
    public String toString() {
      return "TrainingPhase{" +
          "trainingSubject=" + trainingSubject +
          ", orientation=" + orientation +
          '}';
    }
  }

  private class PerformanceWrapper extends TrainableWrapper<SampledTrainable> implements SampledTrainable {

    /**
     * Instantiates a new Performance wrapper.
     *
     * @param trainingSubject the training subject
     */
    public PerformanceWrapper(final SampledTrainable trainingSubject) {
      super(trainingSubject);
    }

    @Nonnull
    @Override
    public SampledCachedTrainable<? extends SampledTrainable> cached() {
      return new SampledCachedTrainable<>(this);
    }

    @Override
    public int getTrainingSize() {
      return getInner().getTrainingSize();
    }

    @Override
    public PointSample measure(final TrainingMonitor monitor) {
      @Nonnull final TimedResult<PointSample> time = TimedResult.time(() ->
          getInner().measure(monitor)
      );
      trainingMeasurementTime.addAndGet(time.timeNanos);
      return time.result;
    }

    @Nonnull
    @Override
    public SampledTrainable setTrainingSize(final int trainingSize) {
      getInner().setTrainingSize(trainingSize);
      return this;
    }

  }

  private class StepResult {
    /**
     * The Current point.
     */
    final PointSample currentPoint;
    /**
     * The Performance.
     */
    final double[] performance;
    /**
     * The Previous.
     */
    final PointSample previous;

    /**
     * Instantiates a new Step result.
     *
     * @param previous     the previous
     * @param currentPoint the current point
     * @param performance  the performance
     */
    public StepResult(final PointSample previous, final PointSample currentPoint, final double[] performance) {
      this.currentPoint = currentPoint;
      this.previous = previous;
      this.performance = performance;
    }

  }
}

