package com.simiacryptus.mindseye.deltas;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.training.TrainingContext;

public class NumberVector implements VectorLogic<NumberVector> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  private final double[] array;

  public <T extends Number> NumberVector(final double[] values) {
    this.array = Arrays.copyOf(values, values.length);
  }

  public <T extends Number> NumberVector(final T[] values) {
    this.array = Arrays.stream(values).mapToDouble(x -> x.doubleValue()).toArray();
  }

  @Override
  public NumberVector add(final NumberVector right) {
    return join(right, (l, r) -> l + r);
  }

  protected double combine(final NumberVector right, final DoubleBinaryOperator joiner) {
    return IntStream.range(0, getArray().length).mapToDouble(i -> {
      final double l = getArray()[i];
      final double r = right.getArray()[i];
      if (Double.isFinite(l) && Double.isFinite(r))
        return joiner.applyAsDouble(l, r);
      else
        return Double.NaN;
    }).sum();
  }

  @Override
  public double dotProduct(final NumberVector right) {
    return combine(right, (l, r) -> l * r);
  }

  public double[] getArray() {
    return this.array;
  }

  public NumberVector getVector(final double fraction) {
    return this;
  }

  public boolean isFrozen() {
    return false;
  }

  protected <T extends Number> NumberVector join(final NumberVector right, final DoubleBinaryOperator joiner) {
    return new NumberVector(IntStream.range(0, getArray().length).mapToDouble(i -> {
      final double l = getArray()[i];
      final double r = right.getArray()[i];
      if (Double.isFinite(l) && Double.isFinite(r))
        return joiner.applyAsDouble(l, r);
      if (Double.isFinite(l))
        return r;
      if (Double.isFinite(r))
        return l;
      return Double.NaN;
    }).toArray());
  }

  @Override
  public double l1() {
    return Arrays.stream(getArray()).map(v -> Math.abs(v)).sum();
  }

  @Override
  public double l2() {
    return Math.sqrt(Arrays.stream(getArray()).map(v -> v * v).sum());
  }

  protected NumberVector map(final DoubleUnaryOperator mapper) {
    return new NumberVector(Arrays.stream(getArray()).map(x -> mapper.applyAsDouble(x)).toArray());
  }

  @Override
  public NumberVector scale(final double f) {
    return map(x -> x * f);
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
