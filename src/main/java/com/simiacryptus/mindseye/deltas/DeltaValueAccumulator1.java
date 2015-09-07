package com.simiacryptus.mindseye.deltas;

import java.util.function.Function;


public class DeltaValueAccumulator1 implements DeltaValueAccumulator<DeltaValueAccumulator1> {
  double sum = 0;

  public DeltaValueAccumulator1() {
  }

  public DeltaValueAccumulator1(final DeltaValueAccumulator1 toCopy) {
    this.sum = this.sum +(toCopy.sum);
  }

  @Override
  public DeltaValueAccumulator1 add(final DeltaValueAccumulator1 r) {
    final DeltaValueAccumulator1 copy = new DeltaValueAccumulator1(this);
    this.sum = this.sum+(r.sum);
    return copy;
  }

  @Override
  public synchronized DeltaValueAccumulator1 add(final double r) {
    this.sum = this.sum+(r);
    return this;
  }

  @Override
  public double doubleValue() {
    return logValue();
  }

  @Override
  public synchronized double logValue() {
    return this.sum;
  }

  private synchronized DeltaValueAccumulator1 map(final Function<Double, Double> f) {
    final DeltaValueAccumulator1 copy = new DeltaValueAccumulator1();
    copy.sum = f.apply(this.sum);
    return copy;
  }

  @Override
  public DeltaValueAccumulator1 multiply(final double r) {
    return map(x -> x*(r));
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(doubleValue());
    return builder.toString();
  }

}
