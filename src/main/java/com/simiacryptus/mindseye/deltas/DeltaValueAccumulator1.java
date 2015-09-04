package com.simiacryptus.mindseye.deltas;

import java.util.function.Function;

import com.simiacryptus.mindseye.math.LogNumber;

public class DeltaValueAccumulator1 implements DeltaValueAccumulator<DeltaValueAccumulator1> {
  LogNumber sum = LogNumber.ZERO;
  
  public DeltaValueAccumulator1() {
  }
  
  public DeltaValueAccumulator1(DeltaValueAccumulator1 toCopy) {
    sum = sum.add(toCopy.sum);
  }
  
  @Override
  public synchronized LogNumber logValue() {
    return sum;
  }
  
  @Override
  public synchronized DeltaValueAccumulator1 add(LogNumber r) {
    sum = sum.add(r);
    return this;
  }
  
  @Override
  public DeltaValueAccumulator1 add(DeltaValueAccumulator1 r) {
    DeltaValueAccumulator1 copy = new DeltaValueAccumulator1(this);
    sum = sum.add(r.sum);
    return copy;
  }
  
  @Override
  public DeltaValueAccumulator1 multiply(double r) {
    return map(x -> x.multiply(r));
  }
  
  private synchronized DeltaValueAccumulator1 map(Function<LogNumber, LogNumber> f) {
    DeltaValueAccumulator1 copy = new DeltaValueAccumulator1();
    copy.sum = f.apply(sum);
    return copy;
  }
  
  @Override
  public double doubleValue() {
    return logValue().doubleValue();
  }
  
}