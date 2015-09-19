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
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class MultiRateGDTrainer implements RateTrainingComponent {

  private static final Logger log = LoggerFactory.getLogger(MultiRateGDTrainer.class);

  private double error = Double.POSITIVE_INFINITY;
  private NDArray[][] masterTrainingData = null;
  private DAGNetwork net = null;
  private double rate = 0.1;
  private double[] rates = null;
  private double temperature = 0.05;
  private boolean verbose = false;

  protected DeltaBuffer calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    final List<NNResult> netresults = eval(trainingContext, data);
    final DeltaBuffer buffer = new DeltaBuffer();
    IntStream.range(0, data.length).parallel().forEach(sample -> {
      final NNResult actualOutput = netresults.get(sample);
      final NDArray delta = new NDArray(new int[] { 1 }, new double[] { -1. }).scale(getRate());
      actualOutput.feedback(delta, buffer);
    });
    return buffer;
  }

  private List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData) {
    return eval(trainingContext, trainingData, true);
  }

  private List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData, final boolean parallel) {
    IntStream stream = IntStream.range(0, trainingData.length);
    if (parallel) {
      stream = stream.parallel();
    }
    return stream.mapToObj(i -> {
      trainingContext.evaluations.increment();
      final NNResult eval = getNet().eval(trainingData[i]);
      return new Tuple2<>(eval, i);
    }).sorted(java.util.Comparator.comparing(x -> x.getSecond())).map(x -> x.getFirst()).collect(Collectors.toList());
  }

  protected ValidationResults evalClassificationValidationData(final TrainingContext trainingContext) {
    return evalClassificationValidationData(trainingContext, getMasterTrainingData());
  }

  protected ValidationResults evalClassificationValidationData(final TrainingContext trainingContext, final NDArray[][] validationSet) {
    final List<NNResult> eval = eval(trainingContext, validationSet);
    final List<NDArray> evalData = eval.stream().map(x -> x.data).collect(Collectors.toList());
    assert validationSet.length == evalData.size();
    final double rms = evalData.stream().parallel().mapToDouble(x -> x.sum()).average().getAsDouble();
    setError(rms);
    return new ValidationResults(evalData, rms);
  }

  @Override
  public synchronized double getError() {
    return this.error;
  }

  public NDArray[][] getMasterTrainingData() {
    return this.masterTrainingData;
  }

  @Override
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

  public DeltaBuffer getVector(final TrainingContext trainingContext) {
    final DeltaBuffer primary = calcDelta(trainingContext, getMasterTrainingData());
    if (isVerbose()) {
      // log.debug(String.format("Primary Delta: %s", primary));
    }
    final DeltaBuffer constraint = calcDelta(trainingContext, getMasterTrainingData()).unitV();
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

  public TrainingComponent setError(final double error) {
    // log.debug(String.format("Error: %s -> %s", this.error, error));
    this.error = error;
    return this;
  }

  public TrainingComponent setMasterTrainingData(final NDArray[][] trainingData) {
    this.masterTrainingData = trainingData;
    return this;
  }

  public TrainingComponent setNet(final DAGNetwork net) {
    this.net = net;
    return this;
  }

  public RateTrainingComponent setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    return this;
  }

  public void setRates(final double[] rates) {
    this.rates = Arrays.copyOf(rates, rates.length);
  }

  public TrainingComponent setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }

  @Override
  public double step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final double prevError = evalClassificationValidationData(trainingContext).rms;
    final double[] rates = getRates();
    if (null == rates)
      return Double.POSITIVE_INFINITY;
    final DeltaBuffer buffer = getVector(trainingContext);
    if (rates.length != buffer.vector().size()) {
      MultiRateGDTrainer.log.debug(String.format("%s != %s", rates.length, buffer.vector().size()));
    }
    assert null != rates && rates.length == buffer.vector().size();
    final List<DeltaFlushBuffer> deltas = buffer.vector();
    assert null != rates && rates.length == deltas.size();
    IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(rates[i]));
    ;
    final double validationError = evalClassificationValidationData(trainingContext).rms;
    if (prevError == validationError) {
      if (this.verbose) {
        MultiRateGDTrainer.log.debug(String.format("Static: (%s)", prevError));
      }
    } else if (!Util.thermalStep(prevError, validationError, getTemperature())) {
      if (this.verbose) {
        MultiRateGDTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
      }
      IntStream.range(0, deltas.size()).forEach(i -> deltas.get(i).write(-rates[i]));
      evalClassificationValidationData(trainingContext);
      return 0.;
    } else {
      if (this.verbose) {
        MultiRateGDTrainer.log.debug(String.format("Validated delta: (%s -> %s) - %s", prevError, validationError, validationError - prevError));
      }
      setError(validationError);
    }
    trainingContext.gradientSteps.increment();
    if (this.verbose) {
      MultiRateGDTrainer.log
          .debug(String.format("Trained Error: %s with rate %s*%s in %.03fs", validationError, getRate(), Arrays.toString(rates), (System.currentTimeMillis() - startMs) / 1000.));
    }
    return validationError - prevError;
  }

  public NDArray[][] getData() {
    return masterTrainingData;
  }

  @Override
  public void refresh() {
    setError(Double.NaN);
  }

}
