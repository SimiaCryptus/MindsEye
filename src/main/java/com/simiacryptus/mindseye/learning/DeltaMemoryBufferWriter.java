package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaMemoryBufferWriter implements DeltaSink, DeltaTransaction {

  private final double[] values;
  private final double[] buffer;

  protected DeltaMemoryBufferWriter() {
    super();
    values = null;
    buffer = null;
  }
  
  public DeltaMemoryBufferWriter(final double[] values) {
    this.values = values;
    buffer = new double[values.length];
  }

  public DeltaMemoryBufferWriter(final NDArray values) {
    this(values.data);
  }

  @Override
  public void feed(final double[] data) {
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.buffer[i] += data[i];
    }
  }

  public void write() {
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.values[i] += buffer[i];
      if(Double.isNaN(this.values[i])) this.values[i] = 0;
      buffer[i] = 0;
    }
  }

  @Override
  public int length() {
    return this.values.length;
  }
  
}
