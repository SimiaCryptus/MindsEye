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
import com.simiacryptus.mindseye.math.NDArray;

import groovy.lang.Tuple2;

public class TrainingContext {

  public static class Counter {

    private final double limit;
    private double value = 0;
    
    public Counter() {
      this(Double.POSITIVE_INFINITY);
    }
    
    public Counter(final double limit) {
      this.limit = limit;
    }

    public double increment() throws TerminationCondition {
      return increment(1.);
    }

    public synchronized double increment(final double delta) throws TerminationCondition {
      this.value += delta;
      if (this.value > this.limit) throw new TerminationCondition(String.format("%s < %s", this.limit, this.value));
      return this.value;
    }
    
    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder();
      builder.append(getClass().getSimpleName());
      builder.append(" [");
      if (Double.isFinite(this.limit)) {
        builder.append("limit=");
        builder.append(this.limit);
        builder.append(", ");
      }
      builder.append("value=");
      builder.append(this.value);
      builder.append("]");
      return builder.toString();
    }
    
  }
  
  @SuppressWarnings("serial")
  public static class TerminationCondition extends RuntimeException {
    
    public TerminationCondition() {
      super();
    }
    
    public TerminationCondition(final String message) {
      super(message);
    }
    
    public TerminationCondition(final String message, final Throwable cause) {
      super(message, cause);
    }
    
    public TerminationCondition(final String message, final Throwable cause, final boolean enableSuppression, final boolean writableStackTrace) {
      super(message, cause, enableSuppression, writableStackTrace);
    }
    
    public TerminationCondition(final Throwable cause) {
      super(cause);
    }

  }

  public static class Timer extends Counter {
    
    public Timer() {
      super();
    }
    
    public Timer(final double limit, final TimeUnit units) {
      super(units.toMillis((long) limit));
    }
    
    public <T> T time(final Supplier<T> f) throws TerminationCondition {
      final long start = System.currentTimeMillis();
      final T retVal = f.get();
      final long elapsed = System.currentTimeMillis() - start;
      increment(elapsed);
      return retVal;
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);
  
  private int[] activeConstraintSet;
  private int[] activeTrainingSet;
  private int[] activeValidationSet;
  public final Counter calibrations;
  public final Counter evaluations;
  public final Counter gradientSteps;
  public final Counter mutations;
  public final Timer overallTimer;
  
  public TrainingContext() {
    this.evaluations = new Counter();
    this.gradientSteps = new Counter();
    this.calibrations = new Counter();
    this.mutations = new Counter();
    this.overallTimer = new Timer();
  }
  
  double calcConstraintSieve(final GradientDescentTrainer inner) {
    final TrainingContext trainingContext = this;
    final NDArray[][] trainingData = inner.getConstraintData(trainingContext);
    final List<NNResult> results = inner.eval(trainingContext, trainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData,
        results.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateConstraintSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getConstraintSet());
  }
  
  public double calcSieves(final GradientDescentTrainer inner) {
    calcConstraintSieve(inner);
    calcTrainingSieve(inner);
    final double validation = calcValidationSieve(inner);
    // log.debug(String.format("Calculated sieves: %s training, %s constraints, %s validation", this.activeTrainingSet.length, this.activeConstraintSet.length,
    // this.activeValidationSet.length));
    return validation;
  }
  
  double calcTrainingSieve(final GradientDescentTrainer inner) {
    final TrainingContext trainingContext = this;
    final NDArray[][] activeTrainingData = inner.getActiveTrainingData(trainingContext);
    final List<NNResult> list = inner.eval(trainingContext, activeTrainingData);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, activeTrainingData,
        list.stream().map(x -> x.data).collect(Collectors.toList()));
    trainingContext.updateTrainingSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getActiveTrainingSet());
  }
  
  double calcValidationSieve(final GradientDescentTrainer current) {
    final TrainingContext trainingContext = this;
    final List<NDArray> result = current.evalValidationData(trainingContext);
    final NDArray[][] trainingData = current.getActiveValidationData(trainingContext);
    final List<Tuple2<Double, Double>> rms = GradientDescentTrainer.stats(trainingContext, trainingData, result);
    trainingContext.updateValidationSieve(rms);
    return DynamicRateTrainer.rms(trainingContext, rms, trainingContext.getActiveValidationSet());
  }
  
  public int[] getActiveTrainingSet() {
    if (null == this.activeTrainingSet) return null;
    if (0 == this.activeTrainingSet.length) return null;
    return this.activeTrainingSet;
  }
  
  public int[] getActiveValidationSet() {
    if (null == this.activeValidationSet) return null;
    if (0 == this.activeValidationSet.length) return null;
    return this.activeValidationSet;
  }
  
  public int[] getConstraintSet() {
    if (null == this.activeConstraintSet) return null;
    if (0 == this.activeConstraintSet.length) return null;
    return this.activeConstraintSet;
  }

  public void setActiveTrainingSet(final int[] activeSet) {
    this.activeTrainingSet = activeSet;
  }
  
  public void setActiveValidationSet(final int[] activeSet) {
    this.activeValidationSet = activeSet;
  }
  
  public void setConstraintSet(final int[] activeSet) {
    this.activeConstraintSet = activeSet;
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("TrainingContext [evaluations=");
    builder.append(this.evaluations);
    builder.append(", calibrations=");
    builder.append(this.calibrations);
    builder.append(", overallTimer=");
    builder.append(this.overallTimer);
    builder.append(", mutations=");
    builder.append(this.mutations);
    builder.append(", gradientSteps=");
    builder.append(this.gradientSteps);
    builder.append("]");
    return builder.toString();
  }
  
  public synchronized int[] updateActiveTrainingSet(final Supplier<int[]> f) {
    this.activeTrainingSet = f.get();
    // if(null == activeTrainingSet) {
    // if(0 == activeTrainingSet.length) activeTrainingSet = null;
    // }
    return this.activeTrainingSet;
  }
  
  public synchronized int[] updateActiveValidationSet(final Supplier<int[]> f) {
    this.activeValidationSet = f.get();
    // if(null == getActiveValidationSet()) {
    // if(0 == activeValidationSet.length) activeValidationSet = null;
    // }
    return this.activeValidationSet;
  }
  
  public synchronized int[] updateConstraintSet(final Supplier<int[]> f) {
    this.activeConstraintSet = f.get();
    // if(null == activeConstraintSet) {
    // if(0 == activeConstraintSet.length) activeConstraintSet = null;
    // }
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