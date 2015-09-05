package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.MultivariateOptimizer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class DynamicRateTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(DynamicRateTrainer.class);
  
  int currentIteration = 0;
  private double temperature = 0.0;
  int generationsSinceImprovement = 0;
  private final GradientDescentTrainer inner = new GradientDescentTrainer();
  int lastCalibratedIteration = Integer.MIN_VALUE;
  final int maxIterations = 1000;
  private double maxRate = 10000;
  double minRate = 0;
  double monteCarloDecayStep = 0.9;
  double monteCarloMin = 0.5;
  private double mutationFactor = 1.;
  double rate = 0.5;
  double[] rates = null;
  private int recalibrationInterval = 10;
  int recalibrationThreshold = 0;
  private double stopError = 0;
  private boolean verbose = false;
  
  public MultivariateFunction asMetaF(final DeltaBuffer lessonVector, TrainingContext trainingContext, GradientDescentTrainer gradientDescentTrainer) {
    double prev = gradientDescentTrainer.getError();
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[lessonVector.map.size()];
      
      @Override
      public double value(final double x[]) {
        final List<DeltaFlushBuffer> writeVectors = trainingContext
            .getNet().getChildren().stream()
            .map(n -> lessonVector.map.get(n))
            .filter(n -> null != n)
            .distinct()
            .sorted(Comparator.comparing(y -> y.getId()))
            .collect(Collectors.toList());
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
  
  protected boolean calibrate(TrainingContext trainingContext) {
    trainingContext.calibrations.increment();
    trainingContext.setActiveTrainingSet(null);
    trainingContext.setActiveValidationSet(null);
    trainingContext.setConstraintSet(new int[] {});
    //trainingContext.calcSieves(getInner());
    GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
    final double prevError = gradientDescentTrainer.calcError(trainingContext, gradientDescentTrainer.evalValidationData(trainingContext));
    boolean inBounds = false;
    PointValuePair optimum;
    try {
      optimum = optimizeRates(trainingContext, gradientDescentTrainer);
      this.rates = DoubleStream.of(optimum.getKey()).map(x -> x * this.rate).toArray();
      inBounds = DoubleStream.of(this.rates).allMatch(r -> getMaxRate() > r)
          && DoubleStream.of(this.rates).anyMatch(r -> this.minRate < r);
      if (inBounds) {
        this.lastCalibratedIteration = this.currentIteration;
        gradientDescentTrainer.step(trainingContext, this.rates);
        final double err = gradientDescentTrainer.calcError(trainingContext, gradientDescentTrainer.evalValidationData(trainingContext));
        final double improvement = prevError - err;
        if (isVerbose()) {
          DynamicRateTrainer.log
              .debug(String.format("Adjusting rates by %s: (%s->%s - %s improvement)", Arrays.toString(this.rates), prevError, err, improvement));
        }
        trainingContext.calcSieves(gradientDescentTrainer);
        return true;
        //return improvement > 0;
      }
    } catch (final Exception e) {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(this.rates),
          gradientDescentTrainer.getError()));
    }
    return false;
  }
  
  public double getTemperature() {
    return this.temperature;
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
  
  public double getMutationFactor() {
    return this.mutationFactor;
  }
  
  public double getRate() {
    return this.rate;
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
  
  public PointValuePair optimizeRates(TrainingContext trainingContext, GradientDescentTrainer current) {
    NDArray[][] validationSet = current.getActiveValidationData(trainingContext);
    List<NDArray> evalValidationData = current.eval(trainingContext,validationSet).stream().map(x1->x1.data).collect(Collectors.toList());
    List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, validationSet, evalValidationData);
    final double prev = rms(trainingContext, rms, null);
    //regenDataSieve(trainingContext);
    
    final DeltaBuffer lessonVector = current.getVector(trainingContext);
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    double fraction = 1.;
    PointValuePair x = null;
    final MultivariateFunction f = asMetaF(lessonVector, trainingContext, current);
    do {
//      trainingContext.setConstraintSet(null);
//      trainingContext.setActiveTrainingSet(null);
//      trainingContext.setActiveValidationSet(null);
      x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(lessonVector.map.size()); // May or may not be cloned before evaluations
      f.value(x.getFirst()); // Leave in optimal state
      fraction *= this.monteCarloDecayStep;
    } while (fraction > this.monteCarloMin && new ArrayRealVector(x.getFirst()).getL1Norm() == 0);
    f.value(new double[lessonVector.map.size()]); // Reset to original state
    evalValidationData = current.eval(trainingContext,validationSet).stream().map(x1->x1.data).collect(Collectors.toList());
    final double calcError = current.calcError(trainingContext, evalValidationData);
    current.setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s",
          Arrays.toString(x.getKey()), x.getValue(), prev, calcError));
    }
    return x;
  }
  
  public static double rms(TrainingContext trainingContext, final List<Tuple2<Double, Double>> rms, int[] activeSet) {
    @SuppressWarnings("resource")
    IntStream stream = null != activeSet ? IntStream.of(activeSet) : IntStream.range(0, rms.size());
    return Math.sqrt(stream
        .filter(i -> i < rms.size())
        .mapToObj(i -> rms.get(i))
        .mapToDouble(x -> x.getSecond())
        .average().getAsDouble());
  }
  
  public DynamicRateTrainer setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
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
  
  public void setMutationFactor(final double mutationRate) {
    this.mutationFactor = mutationRate;
  }
  
  public DynamicRateTrainer setRate(final double rate) {
    this.rate = rate;
    return this;
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
  
  public boolean trainToLocalOptimum(TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      GradientDescentTrainer gradientDescentTrainer = getGradientDescentTrainer();
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
        if(!recalibrateWRetry(trainingContext)) return false;
        this.generationsSinceImprovement = 0;
      }
      final double last = gradientDescentTrainer.getError();
      final double next = gradientDescentTrainer.step(trainingContext, this.rates);
      if (last != next && GradientDescentTrainer.thermalStep(last, next, getTemperature())) {
        this.generationsSinceImprovement = 0;
      } else {
        if (this.recalibrationThreshold < this.generationsSinceImprovement++) {
          if (isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          if(!recalibrateWRetry(trainingContext)) return false;
          this.generationsSinceImprovement = 0;
        }
      }
    }
  }

  public boolean recalibrateWRetry(TrainingContext trainingContext) {
    int retry = 0;
    while (!calibrate(trainingContext)) {
      DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
      if (++retry > 0) return false;
    }
    return true;
  }
  
}
