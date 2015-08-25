package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.math.LogNumber;

public interface DeltaSink {
  public void feed(LogNumber[] data);

  public int length();

}
