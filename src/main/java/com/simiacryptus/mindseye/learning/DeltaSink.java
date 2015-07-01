package com.simiacryptus.mindseye.learning;

public interface DeltaSink {
  public void feed(double[] data);

  public int length();

}
