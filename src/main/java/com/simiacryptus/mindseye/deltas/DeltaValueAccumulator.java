package com.simiacryptus.mindseye.deltas;

import com.simiacryptus.mindseye.math.LogNumber;

public interface DeltaValueAccumulator<T extends DeltaValueAccumulator<T>> {
  
  LogNumber logValue();
  
  T add(LogNumber r);
  
  T add(T r);
  
  T multiply(double r);
  
  double doubleValue();
  
}