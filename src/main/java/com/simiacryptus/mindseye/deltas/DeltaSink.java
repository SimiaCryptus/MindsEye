package com.simiacryptus.mindseye.deltas;

import com.simiacryptus.mindseye.math.LogNumber;

public interface DeltaSink {
  public void feed(LogNumber[] data);

  public int length();

}
