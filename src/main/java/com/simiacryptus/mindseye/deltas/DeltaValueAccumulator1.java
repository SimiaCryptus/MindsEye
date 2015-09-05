package com.simiacryptus.mindseye.deltas;

import java.util.function.Function;

import com.simiacryptus.mindseye.math.LogNumber;

public class DeltaValueAccumulator1 implements DeltaValueAccumulator<DeltaValueAccumulator1> {
  LogNumber sum = LogNumber.ZERO;

  public DeltaValueAccumulator1() {
  }

  public DeltaValueAccumulator1(final DeltaValueAccumulator1 toCopy) {
    this.sum = this.sum.add(toCopy.sum);
  }

  @Override
  public DeltaValueAccumulator1 add(final DeltaValueAccumulator1 r) {
    final DeltaValueAccumulator1 copy = new DeltaValueAccumulator1(this);
    this.sum = this.sum.add(r.sum);
    return copy;
  }

  @Override
  public synchronized DeltaValueAccumulator1 add(final LogNumber r) {
    this.sum = this.sum.add(r);
    return this;
  }

  @Override
  public double doubleValue() {
    return logValue().doubleValue();
  }

  @Override
  public synchronized LogNumber logValue() {
    return this.sum;
  }

  private synchronized DeltaValueAccumulator1 map(final Function<LogNumber, LogNumber> f) {
    final DeltaValueAccumulator1 copy = new DeltaValueAccumulator1();
    copy.sum = f.apply(this.sum);
    return copy;
  }

  @Override
  public DeltaValueAccumulator1 multiply(final double r) {
    return map(x -> x.multiply(r));
  }

}