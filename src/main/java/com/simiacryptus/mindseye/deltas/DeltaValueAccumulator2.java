package com.simiacryptus.mindseye.deltas;

import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Stream;

import com.simiacryptus.mindseye.math.LogNumber;

public class DeltaValueAccumulator2 implements DeltaValueAccumulator<DeltaValueAccumulator2> {
  public TreeSet<LogNumber> numbers = new TreeSet<>();
  
  public DeltaValueAccumulator2() {
  }
  
  public DeltaValueAccumulator2(DeltaValueAccumulator2 toCopy) {
    numbers.addAll(toCopy.numbers);
  }
  
  public synchronized LogNumber logValue() {
    LogNumber[] array = numbers.stream().toArray(i -> new LogNumber[i]);
    if (null == array || 0 == array.length) return LogNumber.ZERO;
    return Stream.of(array).reduce((a, b) -> a.add(b)).get().divide(array.length);
  }
  
  public synchronized DeltaValueAccumulator2 add(LogNumber r) {
    numbers.add(r);
    return this;
  }
  
  public DeltaValueAccumulator2 add(DeltaValueAccumulator2 r) {
    DeltaValueAccumulator2 copy = new DeltaValueAccumulator2(this);
    copy.numbers.addAll(r.numbers);
    return copy;
  }
  
  public DeltaValueAccumulator2 multiply(double r) {
    return map(x -> x.multiply(r));
  }
  
  private synchronized DeltaValueAccumulator2 map(Function<LogNumber, LogNumber> f) {
    DeltaValueAccumulator2 copy = new DeltaValueAccumulator2();
    numbers.stream().map(f).forEach(x -> copy.add(x));
    // copy.sum = f.apply(sum);
    return copy;
  }
  
  public double doubleValue() {
    return logValue().doubleValue();
  }
  
}