package com.simiacryptus.mindseye.training;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaBuffer;
import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork.DAGNode;
import com.simiacryptus.mindseye.net.DAGNetwork.EvaluationContext;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;

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
  private int trainingSize = Integer.MAX_VALUE;
  private DAGNode primaryNode;

  protected DeltaSet calcDelta(final TrainingContext trainingContext, final NDArray[][] data) {
    List<Tuple2<EvaluationContext, Integer>> contexts = initContexts(trainingContext, data, isParallelTraining(), getPrimaryNode());
    return collectVector(getPrimaryNode(), contexts);
  }

  protected DeltaSet collectVector(DAGNode primaryNode, List<Tuple2<EvaluationContext, Integer>> collect) {
    List<NNResult> eval = collect(collect.stream().map(t->new Tuple2<>(primaryNode.get(t.getFirst()), t.getSecond())));
    final NDArray delta = new NDArray(new int[] { 1 }, new double[] { -getRate() });
    final DeltaSet buffer = collectVector(eval, delta);
    return buffer;
  }

  public static DeltaSet collectVector(final List<NNResult> netresults, final NDArray delta) {
    final DeltaSet buffer = new DeltaSet();
    IntStream.range(0, netresults.size()).parallel().mapToObj(sample -> {
      return netresults.get(sample);
    }).forEach(actualOutput->{
      actualOutput.feedback(delta, buffer);
    });
    return buffer;
  }

  private List<NNResult> eval(final TrainingContext trainingContext, final NDArray[][] trainingData, final boolean parallel, DAGNode primaryNode) {
    List<Tuple2<EvaluationContext, Integer>> collect = initContexts(trainingContext, trainingData, parallel, primaryNode);
    Stream<Tuple2<NNResult, Integer>> results = collect.stream().map(t->new Tuple2<>(primaryNode.get(t.getFirst()), t.getSecond()));
    return collect(results);
  }

  public static <T> List<T> collect(Stream<Tuple2<T, Integer>> results) {
    return results.sorted(java.util.Comparator.comparingInt(x -> x.getSecond())).map(x -> x.getFirst()).collect(Collectors.toList());
  }

  public List<Tuple2<EvaluationContext, Integer>> initContexts(final TrainingContext trainingContext, final NDArray[][] trainingData, final boolean parallel, DAGNode primaryNode) {
    DAGNetwork net = getNet();
    IntStream stream = IntStream.range(0, trainingData.length);
    if (parallel) {
      stream = stream.parallel();
    }
    List<Tuple2<EvaluationContext, Integer>> collect = stream.mapToObj(i -> {
      trainingContext.evaluations.increment();
      EvaluationContext exeCtx = net.buildExeCtx(NNLayer.getConstResult(trainingData[i]));
      //primaryNode.get(exeCtx);
      return new Tuple2<>(exeCtx, i);
    }).collect(Collectors.toList());
    return collect;
  }

  private ValidationResults evalClassificationValidationData(final TrainingContext trainingContext, final NDArray[][] validationSet) {
    final List<NNResult> eval = eval(trainingContext, validationSet, isParallelTraining(), getPrimaryNode());
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
    this.setPrimaryNode(net.getHead());
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
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    StepResult result = _step(trainingContext);
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

  public StepResult _step(final TrainingContext trainingContext) {
    NDArray[][] data = getTrainingData();
    final double prevError = evalClassificationValidationData(trainingContext, data).rms;
    final List<DeltaBuffer> deltas = calcDelta(trainingContext, data).vector();
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
  public void reset() {
    setError(Double.NaN);
  }

  public int getTrainingSize() {
    return trainingSize;
  }

  public GradientDescentTrainer setTrainingSize(int trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }

  protected DAGNode getPrimaryNode() {
    return primaryNode;
  }

  protected void setPrimaryNode(DAGNode primaryNode) {
    this.primaryNode = primaryNode;
  }
}
