package com.simiacryptus.mindseye.training;

import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class TrainingContext {
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

  public final Counter evaluations;
  public final Counter calibrations;
  public final Timer overallTimer;
  public final Counter mutations;
  public final Counter gradientSteps;
  
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
  
}