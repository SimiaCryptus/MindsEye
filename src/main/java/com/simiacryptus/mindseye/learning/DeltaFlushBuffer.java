package com.simiacryptus.mindseye.learning;

import java.util.Arrays;

import com.simiacryptus.mindseye.NDArray;

public class DeltaFlushBuffer implements DeltaSink, DeltaTransaction {

  private final DeltaSink values;
  private final double[] buffer;

  protected DeltaFlushBuffer() {
    super();
    values = null;
    buffer = null;
  }
  
  public DeltaFlushBuffer(final DeltaSink values) {
    this.values = values;
    buffer = new double[values.length()];
  }

  public DeltaFlushBuffer(final NDArray values) {
    this(new DeltaMemoryWriter(values.data));
  }

  @Override
  public void feed(final double[] data) {
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.buffer[i] += data[i];
    }
  }

  public void write() {
    this.values.feed(buffer);
    Arrays.fill(buffer, 0);
  }

  @Override
  public int length() {
    return this.values.length();
  }
  
}
