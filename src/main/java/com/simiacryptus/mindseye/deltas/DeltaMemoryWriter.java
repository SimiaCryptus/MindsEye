package com.simiacryptus.mindseye.deltas;

import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaMemoryWriter implements DeltaSink {
  
  private double[] values;
  
  @SuppressWarnings("unused")
  private DeltaMemoryWriter() {
    super();
  }

  public DeltaMemoryWriter(final double[] values) {
    this.values = values;
  }
  
  public DeltaMemoryWriter(final NDArray values) {
    this(values.getData());
  }
  
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
  public void feed(final LogNumber[] data) {
    final int dim = length();
    if (null == data) return;
    for (int i = 0; i < dim; i++) {
      if (null == data[i]) {
        continue;
      }
      this.values[i] += data[i].doubleValue();
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
