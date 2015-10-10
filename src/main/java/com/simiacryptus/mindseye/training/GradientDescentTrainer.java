package com.simiacryptus.mindseye.training;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.DAGNetwork.DAGNode;
import com.simiacryptus.mindseye.net.DAGNetwork.EvaluationContext;

import groovy.lang.Tuple2;

public class GradientDescentTrainer implements RateTrainingComponent {

  public abstract static class StepResult {
    public double finalError;
    public double prevError;

    public StepResult(final double prevError, final double finalError) {
      super();
      this.prevError = prevError;
      this.finalError = finalError;
    }

    public abstract void revert();
  }

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  public static <T> List<T> collectOrderedValues(final Stream<Tuple2<T, Integer>> results) {
    return results
        .sorted(java.util.Comparator.comparingInt(x -> x.getSecond()))
        .map(x -> x.getFirst())
        .collect(Collectors.toList());
  }

  public static DeltaSet collectVector(final List<NNResult> netresults) {
    final DeltaSet buffer = new DeltaSet();
    IntStream.range(0, netresults.size()).parallel().mapToObj(sample -> {
      return netresults.get(sample);
    }).forEach(actualOutput -> {
      actualOutput.accumulate(buffer);
    });
    return buffer;
  }

  private double error = Double.POSITIVE_INFINITY;
  private long hash = Util.R.get().nextLong();
  private DAGNetwork net = null;
  private boolean parallelTraining = true;
  private DAGNode primaryNode;
  private double rate = 0.1;
  private double temperature = 0.0;

  private NDArray[][] trainingData = null;

  private int trainingSize = Integer.MAX_VALUE;

  private boolean verbose = false;

  public StepResult _step(final TrainingContext trainingContext) {
    final NDArray[][] data = getTrainingData();
    if (data.length == 0)
      return new StepResult(Double.NaN, Double.NaN) {
        @Override
        public void revert() {
        }
      };
    final double prevError = evalClassificationValidationData(trainingContext, data).rms;
    final List<DeltaBuffer> deltas = calcDelta(trainingContext, data).vector();
    deltas.stream().forEach(d -> d.write(this.rate));
    final double validationError = evalClassificationValidationData(trainingContext, data).rms;
    return new StepResult(prevError, validationError) {
      @Override
      public void revert() {
        deltas.stream().forEach(d -> d.write(-GradientDescentTrainer.this.rate));
        evalClassificationValidationData(trainingContext, data);
      }
    };
  }

  protected DeltaSet calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    final List<Tuple2<EvaluationContext, Integer>> contexts = initContexts(trainingContext, data, getPrimaryNode());
    return collectVector(getPrimaryNode(), contexts);
  }

  protected DeltaSet collectVector(final DAGNode primaryNode, final List<Tuple2<EvaluationContext, Integer>> collect) {
    final List<NNResult> eval = collectOrderedValues(collect.stream().map(t -> new Tuple2<>(primaryNode.get(t.getFirst()), t.getSecond())));
    return collectVector(eval).scale(-getRate());
  }

  private List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData, final DAGNode primaryNode) {
    final List<Tuple2<EvaluationContext, Integer>> collect = initContexts(trainingContext, trainingData, primaryNode);
    Stream<Tuple2<EvaluationContext, Integer>> stream = collect.stream();
    if (isParallelTraining()) {
      stream = stream.parallel();
    }
    final Stream<Tuple2<NNResult, Integer>> results = stream.map(t -> new Tuple2<>(primaryNode.get(t.getFirst()), t.getSecond()));
    return collectOrderedValues(results);
  }

  private ValidationResults evalClassificationValidationData(final TrainingContext trainingContext, final NDArray[][] validationSet) {
    assert 0 < validationSet.length;
    final List<NNResult> eval = eval(trainingContext, validationSet, getPrimaryNode());
    assert 0 < eval.size();
    final List<NDArray> evalData = eval.stream().map(x -> x.data).collect(Collectors.toList());
    assert 0 < evalData.size();
    assert validationSet.length == evalData.size();
    final double rms = evalData.stream().parallel().mapToDouble(x -> x.sum()).average().getAsDouble();
    setError(rms);
    return new ValidationResults(evalData, rms);
  }

  @Override
  public NDArray[][] getData() {
    return this.trainingData;
    // assert(null!=trainingData);
    // return null==trainingData?new NDArray[][]{}:trainingData;
  }

  @Override
  public synchronized double getError() {
    if (Double.isNaN(this.error))
      return evalClassificationValidationData(new TrainingContext(), getTrainingData()).rms;
    return this.error;
  }

  @Override
  public DAGNetwork getNet() {
    return this.net;
  }

  protected DAGNode getPrimaryNode() {
    return this.primaryNode;
  }

  @Override
  public double getRate() {
    return this.rate;
  }

  public double getTemperature() {
    return this.temperature;
  }

  public NDArray[][] getTrainingData() {

    final NDArray[][] data2 = getData();
    assert 0 < data2.length;
    assert 0 < getTrainingSize();
    final NDArray[][] data = java.util.Arrays.stream(data2).parallel().sorted(java.util.Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash))
        .limit(getTrainingSize()).toArray(i -> new NDArray[i][]);
    return data;
  }

  public int getTrainingSize() {
    return this.trainingSize;
  }

  public List<Tuple2<EvaluationContext, Integer>> initContexts(final TrainingContext trainingContext, final NDArray[][] trainingData, final DAGNode primaryNode) {
    final DAGNetwork net = getNet();
    IntStream stream = IntStream.range(0, trainingData.length);
    if (isParallelTraining()) {
      stream = stream.parallel();
    }
    final List<Tuple2<EvaluationContext, Integer>> collect = stream.mapToObj(i -> {
      trainingContext.evaluations.increment();
      final EvaluationContext exeCtx = net.buildExeCtx(NNLayer.getConstResult(trainingData[i]));
      // primaryNode.get(exeCtx);
      return new Tuple2<>(exeCtx, i);
    }).collect(Collectors.toList());
    return collect;
  }

  public boolean isParallelTraining() {
    return this.parallelTraining;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  @Override
  public void reset() {
    setError(Double.NaN);
  }

  @Override
  public TrainingComponent setData(final NDArray[][] trainingData) {
    this.trainingData = trainingData;
    return this;
  }

  public TrainingComponent setError(final double error) {
    this.error = error;
    return this;
  }

  public TrainingComponent setNet(final DAGNetwork net) {
    this.net = net;
    setPrimaryNode(net.getHead());
    return this;
  }

  public void setParallelTraining(final boolean parallelTraining) {
    this.parallelTraining = parallelTraining;
  }

  protected void setPrimaryNode(final DAGNode primaryNode) {
    this.primaryNode = primaryNode;
  }

  @Override
  public RateTrainingComponent setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    setError(Double.NaN);
    return this;
  }

  public TrainingComponent setTemperature(final double temperature) {
    this.temperature = temperature;
    return this;
  }

  public GradientDescentTrainer setTrainingSize(final int trainingSize) {
    updateHash();
    this.trainingSize = trainingSize;
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
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final StepResult result = _step(trainingContext);
    if (result.prevError == result.finalError) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Static: (%s)", result.prevError));
      }
      setError(result.finalError);
      trainingContext.gradientSteps.increment();
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Trained Error: %s with rate %s in %.03fs", //
            result.finalError, getRate(), (System.currentTimeMillis() - startMs) / 1000.));
      }
      return new TrainingStep(result.prevError, result.finalError, false);
    } else if (!Util.thermalStep(result.prevError, result.finalError, getTemperature())) {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Reverting delta: (%s -> %s) - %s (rate %s)", //
            result.prevError, result.finalError, result.finalError - result.prevError, getRate()));
      }
      result.revert();
      return new TrainingStep(result.prevError, result.finalError, false);
    } else {
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Validated delta: (%s -> %s) - %s", //
            result.prevError, result.finalError, result.finalError - result.prevError));
      }
      setError(result.finalError);
      trainingContext.gradientSteps.increment();
      if (this.verbose) {
        GradientDescentTrainer.log.debug(String.format("Trained Error: %s with rate %s in %.03fs", //
            result.finalError, getRate(), (System.currentTimeMillis() - startMs) / 1000.));
      }
      return new TrainingStep(result.prevError, result.finalError, true);
    }
  }

  private void updateHash() {
    this.hash = Util.R.get().nextLong();
  }
}
