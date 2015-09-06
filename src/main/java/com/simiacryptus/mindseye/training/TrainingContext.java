package com.simiacryptus.mindseye.training;

import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
      if (this.value > this.limit)
        throw new TerminationCondition(String.format("%s < %s", this.limit, this.value));
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

}
