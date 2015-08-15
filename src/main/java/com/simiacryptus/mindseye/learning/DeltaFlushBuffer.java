package com.simiacryptus.mindseye.learning;

import java.util.Arrays;

import com.simiacryptus.mindseye.NDArray;

public class DeltaFlushBuffer implements DeltaSink, DeltaTransaction {

  private final double[] buffer;
  private final DeltaSink inner;
  private double rate = 1.;

  private boolean reset = false;
  
  protected DeltaFlushBuffer() {
    super();
    this.inner = null;
    this.buffer = null;
  }

  public DeltaFlushBuffer(final DeltaSink values) {
    this.inner = values;
    this.buffer = new double[values.length()];
  }

  public DeltaFlushBuffer(final NDArray values) {
    this(new DeltaMemoryWriter(values.getData()));
  }

  @Override
  public void feed(final double[] data) {
    if (this.reset) {
      this.reset = false;
      Arrays.fill(this.buffer, 0);
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.buffer[i] += data[i] * this.rate;
    }
  }

  @Override
  public double getRate() {
    return this.rate;
  }
  
  @Override
  public boolean isFrozen() {
    return false;
  }

  @Override
  public int length() {
    return this.inner.length();
  }

  @Override
  public void setRate(final double rate) {
    this.rate = rate;
  }
  
  @Override
  public void write(final double factor) {
    final double[] cpy = new double[this.buffer.length];
    for (int i = 0; i < this.buffer.length; i++) {
      cpy[i] = this.buffer[i] * factor;
    }
    this.inner.feed(cpy);
    this.reset = true;
  }
  
}
