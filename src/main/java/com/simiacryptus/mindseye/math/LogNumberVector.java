package com.simiacryptus.mindseye.math;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.training.TrainingContext;

@SuppressWarnings({ "unchecked" })
public class LogNumberVector implements VectorLogic<LogNumberVector> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  private final LogNumber[] array;

  public LogNumberVector(final LogNumber[] values) {
    this.array = values;
  }

  @Override
  public LogNumberVector add(final LogNumberVector right) {
    return join(right, (l, r) -> l.add(r));
  }

  @Override
  public double dotProduct(final LogNumberVector right) {
    return sum(right, (l, r) -> l.multiply(r).doubleValue());
  }

  public LogNumber[] getArray() {
    return this.array;
  }

  public LogNumberVector getVector(final double fraction) {
    return this;
  }

  public boolean isFrozen() {
    return false;
  }

  protected <T extends LogNumber> LogNumberVector join(final LogNumberVector right, final BiFunction<T, T, T> joiner) {
    return new LogNumberVector(IntStream.range(0, getArray().length).mapToObj(i -> {
      final T l = (T) getArray()[i];
      final T r = (T) right.getArray()[i];
      if (null != l && null != r)
        return joiner.apply(l, r);
      if (null != l)
        return r;
      if (null != r)
        return l;
      return null;
    }).toArray(i -> new LogNumber[i]));
  }

  @Override
  public double l1() {
    return Math.sqrt(Arrays.stream(getArray()).mapToDouble(v -> {
      final double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }

  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(getArray()).mapToDouble(v -> {
      final double l2 = v.doubleValue();
      return l2 * l2;
    }).sum());
  }

  protected <T extends LogNumber> LogNumberVector map(final Function<T, T> mapper) {
    return new LogNumberVector(Arrays.stream(getArray()).map(x -> mapper.apply((T) x)).toArray(i -> new LogNumber[i]));
  }

  @Override
  public LogNumberVector scale(final double f) {
    return map(x -> x.multiply(f));
  }

  protected <T extends LogNumber> double sum(final LogNumberVector right, final BiFunction<T, T, Double> joiner) {
    return IntStream.range(0, getArray().length).mapToDouble(i -> {
      final T l = (T) getArray()[i];
      final T r = (T) right.getArray()[i];
      if (null != l && null != r)
        return joiner.apply(l, r);
      return 0;
    }).sum();
  }

  public NumberVector toDouble() {
    return new NumberVector(this.array);
    // return new
    // NumberVector(Arrays.stream(this.array).mapToDouble(x->x.doubleValue()).toArray());
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("DeltaFlushBuffer [");
    builder.append(Arrays.toString(getArray()));
    builder.append("]");
    return builder.toString();
  }

}
