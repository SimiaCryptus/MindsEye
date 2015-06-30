package com.simiacryptus.mindseye.learning;

public interface DeltaBuffer {
  public void feed(double[] data);

  public int length();

}
