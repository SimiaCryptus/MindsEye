package com.simiacryptus.mindseye.layers;

public interface DeltaBuffer {
  public int length();
  public void feed(double[] data);

}
