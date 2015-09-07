package com.simiacryptus.mindseye.deltas;

import java.util.TreeSet;
import java.util.function.Function;


public class DeltaValueAccumulator2 implements DeltaValueAccumulator<DeltaValueAccumulator2> {
  public TreeSet<Double> numbers = new TreeSet<>();

  public DeltaValueAccumulator2() {
  }

  public DeltaValueAccumulator2(final DeltaValueAccumulator2 toCopy) {
    this.numbers.addAll(toCopy.numbers);
  }

  @Override
  public DeltaValueAccumulator2 add(final DeltaValueAccumulator2 r) {
    final DeltaValueAccumulator2 copy = new DeltaValueAccumulator2(this);
    copy.numbers.addAll(r.numbers);
    return copy;
  }

  @Override
  public synchronized DeltaValueAccumulator2 add(final double r) {
    this.numbers.add(r);
    return this;
  }

  @Override
  public double doubleValue() {
    return logValue();
  }

  @Override
  public synchronized double logValue() {
    final double[] array = this.numbers.stream().mapToDouble(x->x).toArray();
    if (null == array || 0 == array.length)
      return 0;
    return java.util.stream.DoubleStream.of(array).reduce((a, b) -> a+b).getAsDouble()/array.length;
  }

  private synchronized DeltaValueAccumulator2 map(final Function<Double, Double> f) {
    final DeltaValueAccumulator2 copy = new DeltaValueAccumulator2();
    this.numbers.stream().map(f).forEach(x -> copy.add(x));
    // copy.sum = f.apply(sum);
    return copy;
  }

  @Override
  public DeltaValueAccumulator2 multiply(final double r) {
    return map(x -> x*(r));
  }

}
