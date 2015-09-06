package com.simiacryptus.mindseye.deltas;

import com.simiacryptus.mindseye.math.LogNumber;

public interface DeltaValueAccumulator<T extends DeltaValueAccumulator<T>> {

  T add(LogNumber r);

  T add(T r);

  double doubleValue();

  LogNumber logValue();

  T multiply(double r);

}
