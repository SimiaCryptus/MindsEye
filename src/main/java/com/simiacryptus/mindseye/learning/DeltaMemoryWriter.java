package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.math.NDArray;

public class DeltaMemoryWriter implements DeltaSink {

  private double[] values;

  public DeltaMemoryWriter() {
    super();
  }
  
  public DeltaMemoryWriter(final double[] values) {
    this.values = values;
  }

  public DeltaMemoryWriter(final NDArray values) {
    this(values.getData());
  }

  @Override
  public void feed(final double[] data) {

    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.values[i] += data[i];
      if (!Double.isFinite(this.values[i])) {
        this.values[i] = 0;
      }
    }
  }

  @Override
  public int length() {
    return this.values.length;
  }
  
}
