package com.simiacryptus.mindseye.learning;

import java.util.Arrays;

import com.simiacryptus.mindseye.NDArray;

public class DeltaFlushBuffer implements DeltaSink, DeltaTransaction {

  private final DeltaSink values;
  private final double[] buffer;
  private boolean reset = false;

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
    this(new DeltaMemoryWriter(values.getData()));
  }

  @Override
  public void feed(final double[] data) {
    if (reset) {
      reset = false;
      Arrays.fill(buffer, 0);
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.buffer[i] += data[i] * rate;
    }
  }

  public void write(double factor) {
    double[] cpy = new double[buffer.length];
    for (int i = 0; i < buffer.length; i++) {
      cpy[i] = buffer[i] * factor;
    }
    this.values.feed(cpy);
    reset = true;
  }

  @Override
  public int length() {
    return this.values.length();
  }
  
  public double getRate() {
    return rate;
  }

  public void setRate(double rate) {
    this.rate = rate;
  }

  private double rate = 1.;
  
}
