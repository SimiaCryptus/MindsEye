package com.simiacryptus.mindseye.learning;

public interface DeltaVector {
  double getMobility();

  boolean isFrozen();

  void setRate(double rate);
  
  void write(double factor);
}
