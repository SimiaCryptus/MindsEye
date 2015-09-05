package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class GradientDescentTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);
  
  private int[] activeConstraintSet;
  private int[] activeTrainingSet;
  private int[] activeValidationSet;
  private double error = Double.POSITIVE_INFINITY;
  private NDArray[][] masterTrainingData = null;
  private PipelineNetwork net = null;
  private double rate = 0.3;
  private double temperature = 0.00005;
  private boolean verbose = false;
  
  double calcConstraintSieve(final TrainingContext trainingContext) {
    final NDArray[][] trainingData = getConstraintData(trainingContext);
    final List<NNResult> results = eval(trainingContext, trainingData);
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, trainingData,
        results.stream().map(x -> x.data).collect(Collectors.toList()));
    updateConstraintSieve(rms);
    return Util.rms(trainingContext, rms, getConstraintSet());
  }
  
  public DeltaBuffer calcDelta(final TrainingContext trainingContext, final NDArray[][] activeTrainingData) {
    final List<NNResult> netresults = eval(trainingContext, activeTrainingData);
    final DeltaBuffer buffer = new DeltaBuffer();
    IntStream.range(0, activeTrainingData.length)
        .parallel()
        .forEach(sample -> {
          final NDArray idealOutput = activeTrainingData[sample][1];
          final NNResult actualOutput = netresults.get(sample);
          final NDArray delta = actualOutput.delta(idealOutput);
          final LogNDArray logDelta = delta.log().scale(getRate());
          actualOutput.feedback(logDelta, buffer);
        });
    return buffer;
  }
  
  protected double calcError(final TrainingContext trainingContext, final List<NDArray> results) {
    final NDArray[][] trainingData = getActiveValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, trainingData, results);
    return Util.rms(trainingContext, rms, getActiveValidationSet());
  }
  
  public synchronized double calcSieves(final TrainingContext trainingContext) {
    calcConstraintSieve(trainingContext);
    calcTrainingSieve(trainingContext);
    final double validation = calcValidationSieve(trainingContext);
    // log.debug(String.format("Calculated sieves: %s training, %s constraints, %s validation", this.activeTrainingSet.length, this.activeConstraintSet.length,
    // this.activeValidationSet.length));
    return validation;
  }
  
  double calcTrainingSieve(final TrainingContext trainingContext) {
    final NDArray[][] activeTrainingData = getTrainingData(getActiveTrainingSet());
    final List<NNResult> list = eval(trainingContext, activeTrainingData);
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, activeTrainingData,
        list.stream().map(x -> x.data).collect(Collectors.toList()));
    updateTrainingSieve(rms);
    return Util.rms(trainingContext, rms, getActiveTrainingSet());
  }
  
  double calcValidationSieve(final TrainingContext trainingContext) {
    final List<NDArray> result = evalValidationData(trainingContext);
    final NDArray[][] trainingData = getActiveValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, trainingData, result);
    updateValidationSieve(rms);
    return Util.rms(trainingContext, rms, getActiveValidationSet());
  }
  
  protected List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData) {
    return Stream.of(trainingData)
        .parallel()
        .map(sample -> {
          final NDArray input = sample[0];
          final NDArray output = sample[1];
          trainingContext.evaluations.increment();
          final NNResult eval = getNet().eval(input);
          assert eval.data.dim() == output.dim();
          return eval;
        }).collect(Collectors.toList());
  }
  
  protected List<NDArray> evalValidationData(final TrainingContext trainingContext) {
    final NDArray[][] validationSet = getActiveValidationData(trainingContext);
    final List<NNResult> eval = eval(trainingContext, validationSet);
    return eval.stream().map(x -> x.data).collect(Collectors.toList());
  }
  
  public int[] getActiveTrainingSet() {
    if (null == this.activeTrainingSet) return null;
    if (0 == this.activeTrainingSet.length) return null;
    return this.activeTrainingSet;
  }
  
  public final NDArray[][] getActiveValidationData(final TrainingContext trainingContext) {
    if (null != getActiveValidationSet())
      return IntStream.of(getActiveValidationSet()).mapToObj(i -> getMasterTrainingData()[i]).toArray(i -> new NDArray[i][]);
    return getMasterTrainingData();
  }
  
  public int[] getActiveValidationSet() {
    if (null == this.activeValidationSet) return null;
    if (0 == this.activeValidationSet.length) return null;
    return this.activeValidationSet;
  }
  
  public final NDArray[][] getConstraintData(final TrainingContext trainingContext) {
    return getTrainingData(getConstraintSet());
  }
  
  public int[] getConstraintSet() {
    if (null == this.activeConstraintSet) return null;
    if (0 == this.activeConstraintSet.length) return null;
    return this.activeConstraintSet;
  }
  
  public synchronized double getError() {
    return this.error;
  }
  
  public NDArray[][] getMasterTrainingData() {
    return this.masterTrainingData;
  }
  
  public PipelineNetwork getNet() {
    return this.net;
  }
  
  public double getRate() {
    return this.rate;
  }
  
  public double getTemperature() {
    return this.temperature;
  }
  
  public NDArray[][] getTrainingData(final int[] activeSet) {
    if (null != activeSet) return IntStream.of(activeSet).mapToObj(i -> getMasterTrainingData()[i]).toArray(i -> new NDArray[i][]);
    return getMasterTrainingData();
  }
  
  protected DeltaBuffer getVector(final TrainingContext trainingContext) {
    final DeltaBuffer primary = calcDelta(trainingContext, getTrainingData(getActiveTrainingSet()));
    if (isVerbose()) {
      // log.debug(String.format("Primary Delta: %s", primary));
    }
    final DeltaBuffer constraint = calcDelta(trainingContext, getConstraintData(trainingContext)).unitV();
    if (isVerbose()) {
      // log.debug(String.format("Constraint Delta: %s", constraint));
    }
    final double dotProductConstraint = primary.dotProduct(constraint);
    if (dotProductConstraint < 0) {
      if (isVerbose()) {
        // log.debug(String.format("Removing component: %s", dotProductConstraint));
      }
      return primary.add(constraint.scale(-dotProductConstraint));
    } else {
      if (isVerbose()) {
        // log.debug(String.format("Preserving component: %s", dotProductConstraint));
      }
      return primary;
    }
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public synchronized void setActiveTrainingSet(final int[] activeSet) {
    this.activeTrainingSet = activeSet;
  }
  
  public synchronized void setActiveValidationSet(final int[] activeSet) {
    this.activeValidationSet = activeSet;
  }
  
  public synchronized void setConstraintSet(final int[] activeSet) {
    this.activeConstraintSet = activeSet;
  }
  
  public GradientDescentTrainer setError(final double error) {
    this.error = error;
    return this;
  }
  
  public GradientDescentTrainer setMasterTrainingData(final NDArray[][] trainingData) {
    this.masterTrainingData = trainingData;
    return this;
  }
  
  public GradientDescentTrainer setNet(final PipelineNetwork net) {
    this.net = net;
    return this;
  }
  
  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }
  
  public GradientDescentTrainer setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
  }
  
  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }
  
  public Double step(final TrainingContext trainingContext, final double[] rates) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final double prevError = calcError(trainingContext, evalValidationData(trainingContext));
    setError(prevError);
    if (null == rates) return Double.POSITIVE_INFINITY;
    final DeltaBuffer buffer = getVector(trainingContext);
    assert null != rates && rates.length == buffer.vector().size();
    final List<DeltaFlushBuffer> deltas = buffer.vector();
    assert null != rates && rates.length == deltas.size();
    IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(rates[i]));
    final double validationError = calcError(trainingContext, evalValidationData(trainingContext));
    if (prevError == validationError) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Static: (%s)", prevError));
      }
    } else if (!Util.thermalStep(prevError, validationError, getTemperature())) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
      }
      IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(-rates[i]));
      return prevError;
    } else {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Validated: (%s)", prevError));
      }
      setError(validationError);
    }
    trainingContext.gradientSteps.increment();
    if (this.verbose) {
      GradientDescentTrainer.log.debug(String.format("Trained Error: %s with rate %s*%s in %.03fs",
          validationError, getRate(), Arrays.toString(rates),
          (System.currentTimeMillis() - startMs) / 1000.));
    }
    return validationError - prevError;
  }
  
  public synchronized int[] updateActiveTrainingSet(final Supplier<int[]> f) {
    this.activeTrainingSet = f.get();
    return this.activeTrainingSet;
  }
  
  public synchronized int[] updateActiveValidationSet(final Supplier<int[]> f) {
    this.activeValidationSet = f.get();
    return this.activeValidationSet;
  }
  
  public synchronized int[] updateConstraintSet(final Supplier<int[]> f) {
    this.activeConstraintSet = f.get();
    return this.activeConstraintSet;
  }
  
  public void updateConstraintSieve(final List<Tuple2<Double, Double>> rms) {
    updateConstraintSet(() -> IntStream.range(0, rms.size())
        .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
        .sorted(Comparator.comparing(t -> t.getSecond().getFirst())).limit(50)
        // .filter(t -> t.getSecond().getFirst() > 0.9)
        .mapToInt(t -> t.getFirst()).toArray());
  }
  
  public void updateTrainingSieve(final List<Tuple2<Double, Double>> rms) {
    updateActiveTrainingSet(() -> IntStream.range(0, rms.size())
        .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
        // .filter(t -> t.getSecond().getFirst() < -0.3)
        .filter(t -> 1.8 * Math.random() > -0.5 - t.getSecond().getFirst())
        // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(100)
        .mapToInt(t -> t.getFirst()).toArray());
  }
  
  public void updateValidationSieve(final List<Tuple2<Double, Double>> rms) {
    updateActiveValidationSet(() -> {
      final List<Tuple2<Integer, Tuple2<Double, Double>>> collect = new ArrayList<>(IntStream.range(0, rms.size())
          .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
          .collect(Collectors.toList()));
      Collections.shuffle(collect);
      return collect.stream().limit(400)
          // .filter(t -> t.getSecond().getFirst() < -0.3)
          // .filter(t -> 0.5 * Math.random() > -0. - t.getSecond().getFirst())
          // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(100)
          // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(500)
          .mapToInt(t -> t.getFirst()).toArray();
    });
  }
  
}
