package com.simiacryptus.mindseye.training;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class GradientDescentTrainer {

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  private int[] constraintSet;
  private double error = Double.POSITIVE_INFINITY;
  private NDArray[][] masterTrainingData = null;
  private DAGNetwork net = null;
  private double rate = 0.3;
  private double[] rates = null;
  private double temperature = 0.0;
  private int[] trainingSet;
  private int[] validationSet;
  private boolean verbose = false;

  protected DeltaBuffer calcDelta(final TrainingContext trainingContext, final NDArray[][] activeTrainingData) {
    final List<NNResult> netresults = eval(trainingContext, activeTrainingData);
    final DeltaBuffer buffer = new DeltaBuffer();
    IntStream.range(0, activeTrainingData.length).parallel().forEach(sample -> {
      final NDArray idealOutput = activeTrainingData[sample][1];
      final NNResult actualOutput = netresults.get(sample);
      final NDArray delta = actualOutput.delta(idealOutput);
      final LogNDArray logDelta = delta.log().scale(getRate());
      actualOutput.feedback(logDelta, buffer);
    });
    return buffer;
  }

  protected double calcError(final TrainingContext trainingContext, final List<NDArray> results) {
    final NDArray[][] trainingData = getValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = Util.stats(trainingContext, trainingData, results);
    return Util.rms(trainingContext, rms, null);
  }

  public List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData) {
    return eval(trainingContext, trainingData, true);
  }

  public List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData, final boolean parallel) {
    IntStream stream = IntStream.range(0, trainingData.length);
    if (parallel) {
      stream = stream.parallel();
    }
    return stream.mapToObj(i -> {
      final NDArray[] sample = trainingData[i];
      final NDArray input = sample[0];
      final NDArray output = sample[1];
      trainingContext.evaluations.increment();
      final NNResult eval = getNet().eval(input);
      assert eval.data.dim() == output.dim();
      return new Tuple2<>(eval, i);
    }).sorted(java.util.Comparator.comparing(x -> x.getSecond())).map(x -> x.getFirst()).collect(Collectors.toList());
  }

  protected List<NDArray> evalValidationData(final TrainingContext trainingContext) {
    final NDArray[][] validationSet = getValidationData(trainingContext);
    final List<NNResult> eval = eval(trainingContext, validationSet);
    final List<NDArray> collect = eval.stream().map(x -> x.data).collect(Collectors.toList());
    setError(calcError(trainingContext, collect));
    return collect;
  }

  public final NDArray[][] getConstraintData(final TrainingContext trainingContext) {
    return getTrainingData(getConstraintSet());
  }

  public final NDArray[][] getTrainingData(final TrainingContext trainingContext) {
    return getTrainingData(getTrainingSet());
  }

  public int[] getConstraintSet() {
    if (null == this.constraintSet)
      return null;
    if (0 == this.constraintSet.length)
      return null;
    return this.constraintSet;
  }

  public synchronized double getError() {
    return this.error;
  }

  public NDArray[][] getMasterTrainingData() {
    return this.masterTrainingData;
  }

  public DAGNetwork getNet() {
    return this.net;
  }

  public double getRate() {
    return this.rate;
  }

  public double[] getRates() {
    return this.rates;
  }

  public double getTemperature() {
    return this.temperature;
  }

  public NDArray[][] getTrainingData(final int[] activeSet) {
    if (null != activeSet)
      return IntStream.of(activeSet).mapToObj(i -> getMasterTrainingData()[i]).toArray(i -> new NDArray[i][]);
    return getMasterTrainingData();
  }

  public int[] getTrainingSet() {
    if (null == this.trainingSet)
      return null;
    if (0 == this.trainingSet.length)
      return null;
    return this.trainingSet;
  }

  public final NDArray[][] getValidationData(final TrainingContext trainingContext) {
    if (null != getValidationSet())
      return IntStream.of(getValidationSet()).mapToObj(i -> getMasterTrainingData()[i]).toArray(i -> new NDArray[i][]);
    return getMasterTrainingData();
  }

  public int[] getValidationSet() {
    if (null == this.validationSet)
      return null;
    if (0 == this.validationSet.length)
      return null;
    return this.validationSet;
  }

  public DeltaBuffer getVector(final TrainingContext trainingContext) {
    final DeltaBuffer primary = calcDelta(trainingContext, getTrainingData(getTrainingSet()));
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
        // log.debug(String.format("Removing component: %s",
        // dotProductConstraint));
      }
      return primary.add(constraint.scale(-dotProductConstraint));
    } else {
      if (isVerbose()) {
        // log.debug(String.format("Preserving component: %s",
        // dotProductConstraint));
      }
      return primary;
    }
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public synchronized void setConstraintSet(final int[] activeSet) {
    this.constraintSet = activeSet;
  }

  public GradientDescentTrainer setError(final double error) {
    this.error = error;
    return this;
  }

  public GradientDescentTrainer setMasterTrainingData(final NDArray[][] trainingData) {
    this.masterTrainingData = trainingData;
    return this;
  }

  public GradientDescentTrainer setNet(final DAGNetwork net) {
    this.net = net;
    return this;
  }

  public GradientDescentTrainer setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }

  public void setRates(final double[] rates) {
    this.rates = Arrays.copyOf(rates, rates.length);
  }

  public GradientDescentTrainer setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
  }

  public synchronized void setTrainingSet(final int[] activeSet) {
    this.trainingSet = activeSet;
  }

  public synchronized void setValidationSet(final int[] activeSet) {
    this.validationSet = activeSet;
  }

  public GradientDescentTrainer setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }

  public Double step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    evalValidationData(trainingContext);
    final double prevError = getError();
    final double[] rates = getRates();
    if (null == rates)
      return Double.POSITIVE_INFINITY;
    final DeltaBuffer buffer = getVector(trainingContext);
    assert null != rates && rates.length == buffer.vector().size();
    final List<DeltaFlushBuffer> deltas = buffer.vector();
    assert null != rates && rates.length == deltas.size();
    IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(rates[i]));
    evalValidationData(trainingContext);
    final double validationError = getError();
    if (prevError == validationError) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Static: (%s)", prevError));
      }
    } else if (!Util.thermalStep(prevError, validationError, getTemperature())) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
      }
      IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(-rates[i]));
      evalValidationData(trainingContext);
      return 0.;
    } else {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Validated delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
      }
      setError(validationError);
    }
    trainingContext.gradientSteps.increment();
    if (this.verbose) {
      GradientDescentTrainer.log
          .debug(String.format("Trained Error: %s with rate %s*%s in %.03fs", validationError, getRate(), Arrays.toString(rates), (System.currentTimeMillis() - startMs) / 1000.));
    }
    return validationError - prevError;
  }

}
