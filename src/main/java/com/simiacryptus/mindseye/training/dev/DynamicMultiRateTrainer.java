package com.simiacryptus.mindseye.training.dev;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaBuffer;
import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.dev.MultivariateOptimizer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingComponent;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.training.ValidationResults;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

public class DynamicMultiRateTrainer implements TrainingComponent {

  private static final Logger log = LoggerFactory.getLogger(DynamicMultiRateTrainer.class);

  private int currentIteration = 0;
  private long etaSec = java.util.concurrent.TimeUnit.HOURS.toSeconds(1);
  private int generationsSinceImprovement = 0;
  private final MultiRateGDTrainer inner = new MultiRateGDTrainer();
  private int lastCalibratedIteration = Integer.MIN_VALUE;
  private int maxIterations = 100;
  private double maxRate = 10000;
  private double minRate = 0;
  private int recalibrationInterval = 10;
  private int recalibrationThreshold = 0;
  private double stopError = 1e-2;
  private boolean verbose = false;

  private MultivariateFunction asMetaF(final DeltaSet lessonVector, final TrainingContext trainingContext) {
    final List<DeltaBuffer> vector = lessonVector.vector();
    if (isVerbose()) {
      final String toString = vector.stream().map(x -> x.toString()).reduce((a, b) -> a + "\n\t" + b).get();
      log.debug(String.format("Optimizing delta vector set: \n\t%s", toString));
    }

    final MultiRateGDTrainer gradientDescentTrainer = getGradientDescentTrainer();
    final double prev = gradientDescentTrainer.getError();
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[vector.size()];

      @Override
      public double value(final double x[]) {
        final int layerCount = vector.size();
        final double[] layerRates = Arrays.copyOf(x, layerCount);
        assert layerRates.length == this.pos.length;
        for (int i = 0; i < layerRates.length; i++) {
          final double prev = this.pos[i];
          final double next = layerRates[i];
          final double adj = next - prev;
          vector.get(i).write(adj);
        }
        for (int i = 0; i < layerRates.length; i++) {
          this.pos[i] = layerRates[i];
        }
        final ValidationResults evalValidationData = gradientDescentTrainer.evalClassificationValidationData(trainingContext);
        if (isVerbose()) {
          DynamicMultiRateTrainer.log.debug(String.format("f[%s] = %s (%s)", Arrays.toString(layerRates), evalValidationData.rms, prev - evalValidationData.rms));
        }
        return evalValidationData.rms;
      }
    };
    return f;
  }

  private synchronized boolean calibrate(final TrainingContext trainingContext) {
    synchronized (trainingContext) {
      trainingContext.calibrations.increment();
      final MultiRateGDTrainer gradientDescentTrainer = getGradientDescentTrainer();
      // trainingContext.calcSieves(getInner());
      try {
        final double[] key = optimizeRates(trainingContext);
        if (setRates(trainingContext, key)) {
          this.lastCalibratedIteration = this.currentIteration;
          final double improvement = gradientDescentTrainer.step(trainingContext).improvement();
          if (isVerbose()) {
            DynamicMultiRateTrainer.log.debug(String.format("Adjusting rates by %s: (%s improvement)", Arrays.toString(key), improvement));
          }
          return true;
        }
      } catch (final Exception e) {
        if (isVerbose()) {
          DynamicMultiRateTrainer.log.debug("Error calibrating", e);
        }
      }
      if (isVerbose()) {
        DynamicMultiRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", //
            Arrays.toString(gradientDescentTrainer.getRates()), //
            gradientDescentTrainer.getError()));
      }
      return false;
    }
  }

  @Override
  public double getError() {
    return getGradientDescentTrainer().getError();
  }

  public long getEtaMs() {
    return this.etaSec;
  }

  private MultiRateGDTrainer getGradientDescentTrainer() {
    return this.inner;
  }

  private double getMaxRate() {
    return this.maxRate;
  }

  private double getMinRate() {
    return this.minRate;
  }

  @Override
  public DAGNetwork getNet() {
    return getGradientDescentTrainer().getNet();
  }

  public int getRecalibrationThreshold() {
    return this.recalibrationThreshold;
  }

  public double getStopError() {
    return this.stopError;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  private double[] optimizeRates(final TrainingContext trainingContext) {
    final MultiRateGDTrainer inner = getGradientDescentTrainer();
    final NDArray[][] validationSet = inner.getMasterTrainingData();
    ;
    final double prev = inner.evalClassificationValidationData(trainingContext, validationSet).rms;
    // regenDataSieve(trainingContext);

    final DeltaSet lessonVector = inner.getVector(trainingContext);
    final MultivariateFunction f = asMetaF(lessonVector, trainingContext);
    final int numberOfParameters = lessonVector.vector().size();

    final PointValuePair x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(numberOfParameters);
    f.value(x.getFirst()); // Leave in optimal state
    // f.value(new double[numberOfParameters]); // Reset to original state

    final double calcError = inner.evalClassificationValidationData(trainingContext, validationSet).rms;
    if (this.verbose) {
      DynamicMultiRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s", Arrays.toString(x.getKey()), x.getValue(), prev, calcError));
    }
    return x.getKey();
  }

  private boolean recalibrateWRetry(final TrainingContext trainingContext) {
    int retry = 0;
    while (!calibrate(trainingContext)) {
      DynamicMultiRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
      if (++retry > 0)
        return false;
    }
    return true;
  }

  public TrainingComponent setEtaEnd(final long cnt, final java.util.concurrent.TimeUnit units) {
    this.etaSec = units.toSeconds(cnt);
    return this;
  }

  public TrainingComponent setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }

  public TrainingComponent setMaxRate(final double maxRate) {
    this.maxRate = maxRate;
    return this;
  }

  public TrainingComponent setMinRate(final double minRate) {
    this.minRate = minRate;
    return this;
  }

  private boolean setRates(final TrainingContext trainingContext, final double[] rates) {
    final boolean inBounds = DoubleStream.of(rates).allMatch(r -> getMaxRate() > r) && DoubleStream.of(rates).anyMatch(r -> getMinRate() < r);
    if (inBounds) {
      getGradientDescentTrainer().setRates(rates);
      return true;
    }
    return false;
  }

  public TrainingComponent setRecalibrationThreshold(final int recalibrationThreshold) {
    this.recalibrationThreshold = recalibrationThreshold;
    return this;
  }

  public TrainingComponent setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getGradientDescentTrainer().setVerbose(verbose);
    return this;
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    double prev = getError();
    step2(trainingContext);
    return new TrainingStep(prev, getError(), true);
  }

  private boolean step2(final TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      final TrainingComponent gradientDescentTrainer = getGradientDescentTrainer();
      final double error = gradientDescentTrainer.getError();
      if (getStopError() > error) {
        DynamicMultiRateTrainer.log.debug("Target error reached: " + error);
        return false;
      }
      if (this.maxIterations <= this.currentIteration++) {
        DynamicMultiRateTrainer.log.debug("Maximum recalibrations reached: " + this.currentIteration);
        return false;
      }
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (isVerbose()) {
          DynamicMultiRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
          DynamicMultiRateTrainer.log.debug("Network state: " + getGradientDescentTrainer().getNet().toString());
        }
        if (!recalibrateWRetry(trainingContext))
          return false;
        this.generationsSinceImprovement = 0;
      }
      if (gradientDescentTrainer.step(trainingContext).isChanged()) {
        this.generationsSinceImprovement = 0;
      } else {
        if (getRecalibrationThreshold() < this.generationsSinceImprovement++) {
          if (isVerbose()) {
            DynamicMultiRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
            DynamicMultiRateTrainer.log.debug("Network state: " + getGradientDescentTrainer().getNet().toString());
          }
          if (!recalibrateWRetry(trainingContext))
            return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
  }

  public NDArray[][] getData() {
    return inner.getData();
  }

  @Override
  public void reset() {
    inner.reset();
  }

  public TrainingComponent setData(NDArray[][] data) {
    return inner.setData(data);
  }

}
