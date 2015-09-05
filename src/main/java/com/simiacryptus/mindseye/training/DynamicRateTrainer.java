package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class DynamicRateTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);
  
  int currentIteration = 0;
  int generationsSinceImprovement = 0;
  private final GradientDescentTrainer inner = new GradientDescentTrainer();
  int lastCalibratedIteration = Integer.MIN_VALUE;
  int maxIterations = 100;
  private double maxRate = 10000;
  double minRate = 0;
  double rate = 0.5;
  private double[] rates = null;
  private int recalibrationInterval = 10;
  private int recalibrationThreshold = 0;
  private double stopError = 0;
  
  private boolean verbose = false;
  
  public MultivariateFunction asMetaF(final DeltaBuffer lessonVector, final TrainingContext trainingContext,
      final GradientDescentTrainer gradientDescentTrainer) {
    final double prev = gradientDescentTrainer.getError();
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[lessonVector.vector().size()];
      
      @Override
      public double value(final double x[]) {
        final List<DeltaFlushBuffer> writeVectors = lessonVector.vector();
        final int layerCount = writeVectors.size();
        final double[] layerRates = Arrays.copyOf(x, layerCount);
        // double[] netRates = Arrays.copyOfRange(x, layerCount, current.getCurrentNetworks().size());
        assert layerRates.length == this.pos.length;
        for (int i = 0; i < layerRates.length; i++) {
          final double prev = this.pos[i];
          final double next = layerRates[i];
          final double adj = next - prev;
          writeVectors.get(i).write(adj);
        }
        for (int i = 0; i < layerRates.length; i++) {
          this.pos[i] = layerRates[i];
        }
        final double calcError = gradientDescentTrainer.calcError(trainingContext, gradientDescentTrainer.evalValidationData(trainingContext));
        final double err = Util.geomMean(calcError);
        if (isVerbose()) {
          DynamicRateTrainer.log.debug(String.format("f[%s] = %s (%s; %s)", Arrays.toString(layerRates), err, calcError, prev - calcError));
        }
        return err;
      }
    };
    return f;
  }
  
  protected synchronized boolean calibrate(final TrainingContext trainingContext) {
    synchronized (trainingContext) {
      trainingContext.calibrations.increment();
      getGradientDescentTrainer().setActiveTrainingSet(null);
      getGradientDescentTrainer().setActiveValidationSet(null);
      getGradientDescentTrainer().setConstraintSet(new int[] {});
      // trainingContext.calcSieves(getInner());
      final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
      boolean inBounds = false;
      PointValuePair optimum;
      double[] rates = getRates();
      try {
        optimum = optimizeRates(trainingContext, gradientDescentTrainer);
        rates = DoubleStream.of(optimum.getKey()).map(x -> x * this.rate).toArray();
        inBounds = DoubleStream.of(rates).allMatch(r -> getMaxRate() > r)
            && DoubleStream.of(rates).anyMatch(r -> this.minRate < r);
        if (inBounds) {
          setRates(rates);
          this.lastCalibratedIteration = this.currentIteration;
          final double improvement = -gradientDescentTrainer.step(trainingContext, getRates());
          if (isVerbose()) {
            DynamicRateTrainer.log
                .debug(String.format("Adjusting rates by %s: (%s improvement)", Arrays.toString(rates), improvement));
          }
          getGradientDescentTrainer().calcSieves(trainingContext);
          return true;
          // return improvement > 0;
        }
      } catch (final Exception e) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Error calibrating", e);
        }
      }
      if (isVerbose()) {
        DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(rates),
            gradientDescentTrainer.getError()));
      }
      return false;
    }
  }
  
  public int getGenerationsSinceImprovement() {
    return this.generationsSinceImprovement;
  }
  
  public GradientDescentTrainer getGradientDescentTrainer() {
    return this.inner;
  }
  
  public double getMaxRate() {
    return this.maxRate;
  }
  
  public double getMinRate() {
    return this.minRate;
  }
  
  public double getRate() {
    return this.rate;
  }
  
  public double[] getRates() {
    return this.rates;
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
  
  public PointValuePair optimizeRates(final TrainingContext trainingContext, final GradientDescentTrainer current) {
    final NDArray[][] validationSet = current.getActiveValidationData(trainingContext);
    List<NDArray> evalValidationData = current.eval(trainingContext, validationSet).stream().map(x1 -> x1.data).collect(Collectors.toList());
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, validationSet, evalValidationData);
    final double prev = Util.rms(trainingContext, rms, null);
    // regenDataSieve(trainingContext);
    
    final DeltaBuffer lessonVector = current.getVector(trainingContext);
    if (isVerbose()) {
      log.debug(
          String.format("Optimizing delta vector set: \n\t%s", lessonVector.vector().stream().map(x -> x.toString()).reduce((a, b) -> a + "\n\t" + b).get()));
    }
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    final MultivariateFunction f = asMetaF(lessonVector, trainingContext, current);
    int numberOfParameters = lessonVector.vector().size();
    
    final PointValuePair x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(numberOfParameters); // May or may not be cloned before evaluations
    f.value(x.getFirst()); // Leave in optimal state
    f.value(new double[lessonVector.vector().size()]); // Reset to original state
    evalValidationData = current.eval(trainingContext, validationSet).stream().map(x1 -> x1.data).collect(Collectors.toList());
    final double calcError = current.calcError(trainingContext, evalValidationData);
    current.setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s",
          Arrays.toString(x.getKey()), x.getValue(), prev, calcError));
    }
    return x;
  }
  
  public boolean recalibrateWRetry(final TrainingContext trainingContext) {
    int retry = 0;
    while (!calibrate(trainingContext)) {
      DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
      if (++retry > 0) return false;
    }
    return true;
  }
  
  public DynamicRateTrainer setGenerationsSinceImprovement(final int generationsSinceImprovement) {
    this.generationsSinceImprovement = generationsSinceImprovement;
    return this;
  }
  
  public DynamicRateTrainer setMaxRate(final double maxRate) {
    this.maxRate = maxRate;
    return this;
  }
  
  public DynamicRateTrainer setMinRate(final double minRate) {
    this.minRate = minRate;
    return this;
  }
  
  public DynamicRateTrainer setRate(final double rate) {
    this.rate = rate;
    return this;
  }
  
  public void setRates(final double[] rates) {
    this.rates = rates;
  }
  
  public DynamicRateTrainer setRecalibrationThreshold(final int recalibrationThreshold) {
    this.recalibrationThreshold = recalibrationThreshold;
    return this;
  }
  
  public DynamicRateTrainer setStopError(final double stopError) {
    this.stopError = stopError;
    return this;
  }
  
  public DynamicRateTrainer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    getGradientDescentTrainer().setVerbose(verbose);
    return this;
  }
  
  public boolean trainToLocalOptimum(final TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      final GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
      if (getStopError() > gradientDescentTrainer.getError()) {
        DynamicRateTrainer.log.debug("Target error reached: " + gradientDescentTrainer.getError());
        return false;
      }
      if (this.maxIterations <= this.currentIteration++) {
        DynamicRateTrainer.log.debug("Maximum steps reached");
        return false;
      }
      if (this.lastCalibratedIteration < this.currentIteration - this.recalibrationInterval) {
        if (isVerbose()) {
          DynamicRateTrainer.log.debug("Recalibrating learning rate due to interation schedule at " + this.currentIteration);
        }
        if (!recalibrateWRetry(trainingContext)) return false;
        this.generationsSinceImprovement = 0;
      }
      final double improvement = gradientDescentTrainer.step(trainingContext, getRates());
      if (improvement < 0) {
        this.generationsSinceImprovement = 0;
      } else {
        if (getRecalibrationThreshold() < this.generationsSinceImprovement++) {
          if (isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          if (!recalibrateWRetry(trainingContext)) return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
  }
  
}
