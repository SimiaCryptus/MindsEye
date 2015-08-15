package com.simiacryptus.mindseye.learning;

public interface DeltaTransaction {
  double getRate();

  boolean isFrozen();

  void setRate(double rate);
  
  void write(double factor);
}
