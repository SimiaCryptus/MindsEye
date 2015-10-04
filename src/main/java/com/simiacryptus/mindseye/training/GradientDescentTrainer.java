package com.simiacryptus.mindseye.training;

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
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class GradientDescentTrainer implements RateTrainingComponent {

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  private double error = Double.POSITIVE_INFINITY;
  private NDArray[][] trainingData = null;
  private DAGNetwork net = null;
  private boolean parallelTraining = true;
  private double rate = 0.1;
  private double temperature = 0.0;
  private boolean verbose = false;
  private long hash = Util.R.get().nextLong();
  private int trainingSize = 300;

  private DeltaBuffer calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    final List<NNResult> netresults = eval(trainingContext, data, isParallelTraining());
    final DeltaBuffer buffer = new DeltaBuffer();
    IntStream.range(0, data.length).parallel().forEach(sample -> {
      final NNResult actualOutput = netresults.get(sample);
      final NDArray delta = new NDArray(new int[] { 1 }, new double[] { -1. }).scale(getRate());
      actualOutput.feedback(delta, buffer);
    });
    return buffer;
  }

  private List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData, final boolean parallel) {
    IntStream stream = IntStream.range(0, trainingData.length);
    if (parallel) {
      stream = stream.parallel();
    }
    return stream.mapToObj(i -> {
      trainingContext.evaluations.increment();
      final NNResult eval = getNet().eval(new EvaluationContext(), trainingData[i]);
      return new Tuple2<>(eval, i);
    }).sorted(java.util.Comparator.comparing(x -> x.getSecond())).map(x -> x.getFirst()).collect(Collectors.toList());
  }

  private ValidationResults evalClassificationValidationData(final TrainingContext trainingContext, final NDArray[][] validationSet) {
    final List<NNResult> eval = eval(trainingContext, validationSet, isParallelTraining());
    final List<NDArray> evalData = eval.stream().map(x -> x.data).collect(Collectors.toList());
    assert validationSet.length == evalData.size();
    final double rms = evalData.stream().parallel().mapToDouble(x -> x.sum()).average().getAsDouble();
    setError(rms);
    return new ValidationResults(evalData, rms);
  }

  @Override
  public synchronized double getError() {
    if(Double.isNaN(error)){
      return evalClassificationValidationData(new TrainingContext(), getTrainingData()).rms;
    }
    return this.error;
  }

  @Override
  public DAGNetwork getNet() {
    return this.net;
  }

  @Override
  public double getRate() {
    return this.rate;
  }

  public double getTemperature() {
    return this.temperature;
  }

  private DeltaBuffer getVector(final TrainingContext trainingContext, NDArray[][] data) {
    final DeltaBuffer primary = calcDelta(trainingContext, data);
    if (isVerbose()) {
      // log.debug(String.format("Primary Delta: %s", primary));
    }
//    final DeltaBuffer constraint = calcDelta(trainingContext, getData()).unitV();
//    if (isVerbose()) {
//      // log.debug(String.format("Constraint Delta: %s", constraint));
//    }
//    final double dotProductConstraint = primary.dotProduct(constraint);
//    if (dotProductConstraint < 0) {
//      if (isVerbose()) {
//        // log.debug(String.format("Removing component: %s", dotProductConstraint));
//      }
//      return primary.add(constraint.scale(-dotProductConstraint));
//    }
    if (isVerbose()) {
      // log.debug(String.format("Preserving component: %s", dotProductConstraint));
    }
    return primary;
  }

  public boolean isParallelTraining() {
    return this.parallelTraining;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public TrainingComponent setError(final double error) {
    this.error = error;
    return this;
  }

  public TrainingComponent setData(final NDArray[][] trainingData) {
    this.trainingData = trainingData;
    return this;
  }

  public TrainingComponent setNet(final DAGNetwork net) {
    this.net = net;
    return this;
  }

  public void setParallelTraining(final boolean parallelTraining) {
    this.parallelTraining = parallelTraining;
  }

  public RateTrainingComponent setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    updateHash();
    setError(Double.NaN);
    return this;
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

  public abstract static class StepResult {
    public double prevError;
    public double finalError;
    public StepResult(double prevError, double finalError) {
      super();
      this.prevError = prevError;
      this.finalError = finalError;
    }
    public abstract void revert();
  }
  
  @Override
  public double step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    StepResult result = _step(trainingContext);
    if (result.prevError == result.finalError) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Static: (%s)", result.prevError));
      }
    } else if (!Util.thermalStep(result.prevError, result.finalError, getTemperature())) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s (rate %s)", //
            result.prevError, result.finalError, result.finalError - result.prevError, getRate()));
      }
      result.revert();
      return result.prevError;
    } else {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Validated delta: (%s -> %s) - %s", //
            result.prevError, result.finalError, result.finalError - result.prevError));
      }
      setError(result.finalError);
    }
    trainingContext.gradientSteps.increment();
    if (this.verbose) {
      GradientDescentTrainer.log.debug(String.format("Trained Error: %s with rate %s in %.03fs", //
          result.finalError, getRate(), (System.currentTimeMillis() - startMs) / 1000.));
    }
    
    return result.finalError;
  }

  public StepResult _step(final TrainingContext trainingContext) {
    NDArray[][] data = getTrainingData();
    final double prevError = evalClassificationValidationData(trainingContext, data).rms;
    final List<DeltaFlushBuffer> deltas = getVector(trainingContext, data).vector();
    deltas.stream().forEach(d->d.write(rate));
    final double validationError = evalClassificationValidationData(trainingContext, data).rms;
    return new StepResult(prevError,validationError){
      @Override
      public void revert() {
        deltas.stream().forEach(d->d.write(-rate));
        evalClassificationValidationData(trainingContext, data);
      }
    };
  }
  
  public NDArray[][] getTrainingData() {
    NDArray[][] data = java.util.Arrays.stream(getData())
        .parallel()
        .sorted(java.util.Comparator.comparingLong(y->System.identityHashCode(y) ^ hash))
        .limit(getTrainingSize())
        .toArray(i->new NDArray[i][]);
    return data;
  }

  private void updateHash() {
    hash = Util.R.get().nextLong();
  }

  public NDArray[][] getData() {
    return trainingData;
  }

  @Override
  public void refresh() {
    setError(Double.NaN);
  }

  public int getTrainingSize() {
    return trainingSize;
  }

  public GradientDescentTrainer setTrainingSize(int trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }

}
