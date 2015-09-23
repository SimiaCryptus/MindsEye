package com.simiacryptus.mindseye.deltas;

public class DeltaValueAccumulator1 implements DeltaValueAccumulator<DeltaValueAccumulator1> {
  double sum = 0;

  public DeltaValueAccumulator1() {
  }

  public DeltaValueAccumulator1(final DeltaValueAccumulator1 toCopy) {
    this.sum = toCopy.sum;
  }

  @Override
  public DeltaValueAccumulator1 add(final DeltaValueAccumulator1 r) {
    this.sum += r.sum;
    return this;
  }

  @Override
  public final DeltaValueAccumulator1 add(final double r) {
    this.sum = this.sum + r;
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

  @Override
  public DeltaValueAccumulator1 multiply(final double r) {
    this.sum *= r;
    return this;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(doubleValue());
    return builder.toString();
  }

}
