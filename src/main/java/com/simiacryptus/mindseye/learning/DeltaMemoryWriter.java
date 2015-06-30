package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaMemoryWriter implements DeltaBuffer {

  private double[] values;

  public DeltaMemoryWriter() {
    super();
  }
  
  public DeltaMemoryWriter(final double[] values) {
    this.values = values;
  }

  public DeltaMemoryWriter(final NDArray values) {
    this(values.data);
  }

  @Override
  public void feed(final double[] data) {

    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.values[i] += data[i];
    }
  }

  @Override
  public int length() {
    return this.values.length;
  }
  
}
