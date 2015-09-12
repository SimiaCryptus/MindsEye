package com.simiacryptus.mindseye.deltas;


public interface DeltaValueAccumulator<T extends DeltaValueAccumulator<T>> {

  T add(double r);

  T add(T r);

  double doubleValue();

  T multiply(double r);

  double logValue();

}
