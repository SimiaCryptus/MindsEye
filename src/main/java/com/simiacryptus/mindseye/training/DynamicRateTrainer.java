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
import com.simiacryptus.mindseye.layers.NNLayer;
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
  private GradientDescentTrainer inner;
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
  
  public DynamicRateTrainer() {
    this(new GradientDescentTrainer());
  }
  
  public DynamicRateTrainer(final GradientDescentTrainer gradientDescentTrainer) {
    this.setInner(gradientDescentTrainer);
  }
  
  public MultivariateFunction asMetaF(final DeltaBuffer lessonVector, final double fraction, TrainingContext trainingContext) {
    double prev = error(trainingContext);
    final MultivariateFunction f = new MultivariateFunction() {
      double[] pos = new double[lessonVector.map.size()];
      
      @Override
      public double value(final double x[]) {
        final GradientDescentTrainer current = DynamicRateTrainer.this.getInner();
        final List<DeltaFlushBuffer> writeVectors = current
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
        final double calcError = current.calcError(trainingContext, current.evalValidationData(trainingContext));
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
    final double prevError = getInner().calcError(trainingContext, getInner().evalValidationData(trainingContext));
    boolean inBounds = false;
    PointValuePair optimum;
    try {
      optimum = optimizeRates(trainingContext);
      getInner()
          .getNet().getChildren().stream().distinct()
          .forEach(layer -> layer.setStatus(optimum.getValue()));
      this.rates = DoubleStream.of(optimum.getKey()).map(x -> x * this.rate).toArray();
      inBounds = DoubleStream.of(this.rates).allMatch(r -> getMaxRate() > r)
          && DoubleStream.of(this.rates).anyMatch(r -> this.minRate < r);
      if (inBounds) {
        this.lastCalibratedIteration = this.currentIteration;
        trainOnce(trainingContext);
        final double err = getInner().calcError(trainingContext, getInner().evalValidationData(trainingContext));
        final double improvement = prevError - err;
        if (isVerbose()) {
          DynamicRateTrainer.log
              .debug(String.format("Adjusting rates by %s: (%s->%s - %s improvement)", Arrays.toString(this.rates), prevError, err, improvement));
        }
        // return true;
        boolean improved = improvement > 0;
        regenDataSieve(trainingContext);
        return improved;
      }
    } catch (final Exception e) {
      if (isVerbose()) {
        DynamicRateTrainer.log.debug("Error calibrating", e);
      }
    }
    if (isVerbose()) {
      DynamicRateTrainer.log.debug(String.format("Calibration rejected at %s with %s error", Arrays.toString(this.rates),
          getInner().getError()));
    }
    return false;
  }
  
  public double error(TrainingContext trainingContext) {
    return getInner().getError();
  }
  
  public double getTemperature() {
    return this.temperature;
  }
  
  public int getGenerationsSinceImprovement() {
    return this.generationsSinceImprovement;
  }
  
  public GradientDescentTrainer getInner() {
    return this.inner;
  }
  
  public List<NNLayer> getLayers() {
    return getInner().getLayers();
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
  
  public PipelineNetwork getNetwork() {
    return this.getInner().getNet();
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
  
  public synchronized PointValuePair optimizeRates(TrainingContext trainingContext) {
    GradientDescentTrainer current = getInner();
    NDArray[][] validationSet = current.getActiveValidationData(trainingContext);
    List<NNResult> evalValidation = current.eval(trainingContext,validationSet);
    List<NDArray> evalValidationData = evalValidation.stream().map(x1->x1.data).collect(Collectors.toList());
    List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, validationSet, evalValidationData);
    final double prev = rms(trainingContext, rms, null);
    //regenDataSieve(trainingContext);
    
    final DeltaBuffer lessonVector = current.prelearn(trainingContext);
    // final double[] one = DoubleStream.generate(() -> 1.).limit(dims).toArray();
    double fraction = 1.;
    PointValuePair x = null;
    final MultivariateFunction f = asMetaF(lessonVector, fraction, trainingContext);
    do {
//      trainingContext.setConstraintSet(null);
//      trainingContext.setActiveTrainingSet(null);
//      trainingContext.setActiveValidationSet(null);
      x = new MultivariateOptimizer(f).setMaxRate(getMaxRate()).minimize(lessonVector.map.size()); // May or may not be cloned before evaluations
      f.value(x.getFirst()); // Leave in optimal state
      fraction *= this.monteCarloDecayStep;
    } while (fraction > this.monteCarloMin && new ArrayRealVector(x.getFirst()).getL1Norm() == 0);
    // f.value(new double[lessonVector.map.size()]); // Reset to original state
    final double calcError = current.calcError(trainingContext, evalValidationData);
    current.setError(calcError);
    if (this.verbose) {
      DynamicRateTrainer.log.debug(String.format("Terminated search at position: %s (%s), error %s->%s",
          Arrays.toString(x.getKey()), x.getValue(),
          (prev), calcError));
    }
    return x;
  }
  
  public double regenDataSieve(TrainingContext trainingContext) {
    calcConstraintSieve(trainingContext);
    calcTrainingSieve(trainingContext);
    return calcValidationSieve(trainingContext);
  }

  protected double calcConstraintSieve(TrainingContext trainingContext) {
    GradientDescentTrainer inner = getInner();
    NDArray[][] trainingData = inner.getConstraintData(trainingContext);
    final List<NNResult> results = inner.eval(trainingContext, trainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData, results.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateConstraintSieve(rms);
    return rms(trainingContext, rms, trainingContext.getConstraintSet());
  }

  protected double calcTrainingSieve(TrainingContext trainingContext) {
    GradientDescentTrainer inner = getInner();
    NDArray[][] activeTrainingData = inner.getActiveTrainingData(trainingContext);
    final List<NNResult> list = inner.eval(trainingContext, activeTrainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, activeTrainingData, list.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateTrainingSieve(rms);
    return rms(trainingContext, rms, trainingContext.getActiveTrainingSet());
  }
  
  protected double calcValidationSieve(TrainingContext trainingContext) {
    GradientDescentTrainer current = getInner();
    final List<NDArray> result = current.evalValidationData(trainingContext);
    final NDArray[][] trainingData = getInner().getActiveValidationData(trainingContext);
    getInner();
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData, result);
    trainingContext.updateValidationSieve(rms);
    return rms(trainingContext, rms, trainingContext.getActiveValidationSet());
  }

  static double rms(TrainingContext trainingContext, final List<Tuple2<Double, Double>> rms, int[] activeSet) {
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
    getInner().setVerbose(verbose);
    return this;
  }
  
  public double trainOnce(TrainingContext trainingContext) throws TerminationCondition {
    getInner().step(trainingContext, this.rates);
    final double error = error(trainingContext);
    getInner().getNet().getChildren().stream()
        .distinct()
        .forEach(layer -> layer.setStatus(error));
    return error;
  }
  
  public boolean trainToLocalOptimum(TrainingContext trainingContext) throws TerminationCondition {
    this.currentIteration = 0;
    this.generationsSinceImprovement = 0;
    this.lastCalibratedIteration = Integer.MIN_VALUE;
    while (true) {
      if (getStopError() > error(trainingContext)) {
        DynamicRateTrainer.log.debug("Target error reached: " + error(trainingContext));
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
        int retry = 0;
        while (!calibrate(trainingContext)) {
          DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
          if (++retry > 0) return false;
        }
      }
      final double last = getInner().getError();
      final double next = trainOnce(trainingContext);
      if (last != next && GradientDescentTrainer.thermalStep(last, next, getTemperature())) {
        this.generationsSinceImprovement = 0;
      } else {
        if (this.recalibrationThreshold < this.generationsSinceImprovement++) {
          if (isVerbose()) {
            DynamicRateTrainer.log.debug("Recalibrating learning rate due to non-descending step");
          }
          if (!calibrate(trainingContext)) {
            DynamicRateTrainer.log.debug("Failed recalibration at iteration " + this.currentIteration);
            return false;
          } else {
            this.generationsSinceImprovement = 0;
          }
        }
      }
    }
  }
  
  public DynamicRateTrainer setInner(GradientDescentTrainer inner) {
    this.inner = inner;
    return this;
  }
  
}
