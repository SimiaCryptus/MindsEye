package com.simiacryptus.mindseye.learning;

public interface DeltaTransaction {
  double getRate();
  
  void setRate(double rate);
  
  void write(double factor);

  boolean isFrozen();
}
