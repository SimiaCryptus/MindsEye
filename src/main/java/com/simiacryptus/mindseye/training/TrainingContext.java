package com.simiacryptus.mindseye.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;

import groovy.lang.Tuple2;

public class TrainingContext {
  
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  @SuppressWarnings("serial")
  public static class TerminationCondition extends RuntimeException
  {

    public TerminationCondition() {
      super();
    }

    public TerminationCondition(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
      super(message, cause, enableSuppression, writableStackTrace);
    }

    public TerminationCondition(String message, Throwable cause) {
      super(message, cause);
    }

    public TerminationCondition(String message) {
      super(message);
    }

    public TerminationCondition(Throwable cause) {
      super(cause);
    }
    
  }
  
  public static class Counter {
    
    private final double limit;
    private double value = 0;

    public Counter() {
      this(Double.POSITIVE_INFINITY);
    }

    public Counter(double limit) {
      this.limit = limit;
    }
    
    public double increment() throws TerminationCondition {
      return increment(1.);
    }
    
    public synchronized double increment(double delta) throws TerminationCondition {
      value += delta;
      if(value > limit) throw new TerminationCondition(String.format("%s < %s", limit, value));
      return value;
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append(getClass().getSimpleName());
      builder.append(" [");
      if (Double.isFinite(limit)) {
        builder.append("limit=");
        builder.append(limit);
        builder.append(", ");
      }
      builder.append("value=");
      builder.append(value);
      builder.append("]");
      return builder.toString();
    }

  }
  
  public static class Timer extends Counter {

    public Timer() {
      super();
    }

    public Timer(double limit, TimeUnit units) {
      super(units.toMillis((long) limit));
    }

    public <T> T time(Supplier<T> f) throws TerminationCondition {
      long start = System.currentTimeMillis();
      T retVal = f.get();
      long elapsed = System.currentTimeMillis() - start;
      increment(elapsed);
      return retVal;
    }
    
  }

  private PipelineNetwork net = null;
  public final Counter evaluations;
  public final Counter calibrations;
  public final Timer overallTimer;
  public final Counter mutations;
  public final Counter gradientSteps;
  private int[] activeTrainingSet;
  private int[] activeValidationSet;
  private int[] activeConstraintSet;

  public TrainingContext() {
    this.evaluations = new Counter();
    this.gradientSteps = new Counter();
    this.calibrations = new Counter();
    this.mutations = new Counter();
    this.overallTimer = new Timer();
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("TrainingContext [evaluations=");
    builder.append(evaluations);
    builder.append(", calibrations=");
    builder.append(calibrations);
    builder.append(", overallTimer=");
    builder.append(overallTimer);
    builder.append(", mutations=");
    builder.append(mutations);
    builder.append(", gradientSteps=");
    builder.append(gradientSteps);
    builder.append("]");
    return builder.toString();
  }

  public int[] getActiveTrainingSet() {
    if(null == activeTrainingSet) return null;
    if(0 == activeTrainingSet.length) return null;
    return activeTrainingSet;
  }

  public int[] getConstraintSet() {
    if(null == activeConstraintSet) return null;
    if(0 == activeConstraintSet.length) return null;
    return activeConstraintSet;
  }

  public void setConstraintSet(int[] activeSet) {
    this.activeConstraintSet = activeSet;
  }

  public synchronized int[] updateConstraintSet(Supplier<int[]> f) {
    activeConstraintSet = f.get();
//    if(null == activeConstraintSet) {
//      if(0 == activeConstraintSet.length) activeConstraintSet = null;
//    }
    return activeConstraintSet;
  }
  public void setActiveTrainingSet(int[] activeSet) {
    this.activeTrainingSet = activeSet;
  }

  public synchronized int[] updateActiveTrainingSet(Supplier<int[]> f) {
    activeTrainingSet = f.get();
//    if(null == activeTrainingSet) {
//      if(0 == activeTrainingSet.length) activeTrainingSet = null;
//    }
    return activeTrainingSet;
  }
  
  public int[] getActiveValidationSet() {
    if(null == activeValidationSet) return null;
    if(0 == activeValidationSet.length) return null;
    return activeValidationSet;
  }

  public void setActiveValidationSet(int[] activeSet) {
    this.activeValidationSet = activeSet;
  }

  public synchronized int[] updateActiveValidationSet(Supplier<int[]> f) {
    activeValidationSet = f.get();
//    if(null == getActiveValidationSet()) {
//      if(0 == activeValidationSet.length) activeValidationSet = null;
//    }
    return activeValidationSet;
  }

  public void updateTrainingSieve(final List<Tuple2<Double, Double>> rms) {
    this.updateActiveTrainingSet(() -> IntStream.range(0, rms.size())
        .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
        // .filter(t -> t.getSecond().getFirst() < -0.3)
        .filter(t -> 1.8 * Math.random() > -0.5 - t.getSecond().getFirst())
        //.sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(100)
        .mapToInt(t -> t.getFirst()).toArray());
  }

  public void updateConstraintSieve(final List<Tuple2<Double, Double>> rms) {
    this.updateConstraintSet(() -> IntStream.range(0, rms.size())
        .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
        .sorted(Comparator.comparing(t -> t.getSecond().getFirst())).limit(50)
        //.filter(t -> t.getSecond().getFirst() > 0.9)
        .mapToInt(t -> t.getFirst()).toArray());
  }

  public void updateValidationSieve(final List<Tuple2<Double, Double>> rms) {
    this.updateActiveValidationSet(() -> {
      List<Tuple2<Integer, Tuple2<Double, Double>>> collect = new ArrayList<>(IntStream.range(0, rms.size())
          .mapToObj(i -> new Tuple2<>(i, rms.get(0)))
          .collect(Collectors.toList()));
      Collections.shuffle(collect);
      return collect.stream().limit(400)
          // .filter(t -> t.getSecond().getFirst() < -0.3)
          //.filter(t -> 0.5 * Math.random() > -0. - t.getSecond().getFirst())
          //.sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(100)
          // .sorted(Comparator.comparing(t -> -t.getSecond().getFirst())).limit(500)
          .mapToInt(t -> t.getFirst()).toArray();
    });
  }

  double calcConstraintSieve(GradientDescentTrainer inner) {
    TrainingContext trainingContext = this;
    NDArray[][] trainingData = inner.getConstraintData(trainingContext);
    final List<NNResult> results = inner.eval(trainingContext, trainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData, results.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateConstraintSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getConstraintSet());
  }

  double calcTrainingSieve(GradientDescentTrainer inner) {
    TrainingContext trainingContext = this;
    NDArray[][] activeTrainingData = inner.getActiveTrainingData(trainingContext);
    final List<NNResult> list = inner.eval(trainingContext, activeTrainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, activeTrainingData, list.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateTrainingSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getActiveTrainingSet());
  }
  
  double calcValidationSieve(GradientDescentTrainer current) {
    TrainingContext trainingContext = this;
    final List<NDArray> result = current.evalValidationData(trainingContext);
    final NDArray[][] trainingData = current.getActiveValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData, result);
    trainingContext.updateValidationSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getActiveValidationSet());
  }

  public double calcSieves(GradientDescentTrainer inner) {
    calcConstraintSieve(inner);
    calcTrainingSieve(inner);
    double validation = calcValidationSieve(inner);
    //log.debug(String.format("Calculated sieves: %s training, %s constraints, %s validation", this.activeTrainingSet.length, this.activeConstraintSet.length, this.activeValidationSet.length));
    return validation;
  }

  public PipelineNetwork getNet() {
    return net;
  }

  public TrainingContext setNet(PipelineNetwork net) {
    this.net = net;
    return this;
  }


}