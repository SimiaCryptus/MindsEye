package com.simiacryptus.mindseye.learning;

public interface DeltaBuffer {
  public int length();
  public void feed(double[] data);

}
