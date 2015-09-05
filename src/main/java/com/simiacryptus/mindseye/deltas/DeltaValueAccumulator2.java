package com.simiacryptus.mindseye.deltas;

import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.math.LogNumber;

public class DeltaValueAccumulator2 implements DeltaValueAccumulator<DeltaValueAccumulator2> {
  public TreeSet<LogNumber> numbers = new TreeSet<>();
  
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
  public synchronized DeltaValueAccumulator2 add(final LogNumber r) {
    this.numbers.add(r);
    return this;
  }
  
  @Override
  public double doubleValue() {
    return logValue().doubleValue();
  }
  
  @Override
  public synchronized LogNumber logValue() {
    final LogNumber[] array = this.numbers.stream().toArray(i -> new LogNumber[i]);
    if (null == array || 0 == array.length) return LogNumber.ZERO;
    return Stream.of(array).reduce((a, b) -> a.add(b)).get().divide(array.length);
  }
  
  private synchronized DeltaValueAccumulator2 map(final Function<LogNumber, LogNumber> f) {
    final DeltaValueAccumulator2 copy = new DeltaValueAccumulator2();
    this.numbers.stream().map(f).forEach(x -> copy.add(x));
    // copy.sum = f.apply(sum);
    return copy;
  }
  
  @Override
  public DeltaValueAccumulator2 multiply(final double r) {
    return map(x -> x.multiply(r));
  }
  
}