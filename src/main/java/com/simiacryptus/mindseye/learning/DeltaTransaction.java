package com.simiacryptus.mindseye.learning;

public interface DeltaTransaction {
  void write(double factor);
  void setRate(double rate);
  double getRate();
}
