package com.simiacryptus.mindseye.learning;

import java.util.Random;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaSampler implements DeltaSink {
  private double sampling = 2.;
  private DeltaSink values;

  public DeltaSampler() {
    super();
  }

  public DeltaSampler(final DeltaSink values) {
    this.values = values;
  }

  public DeltaSampler(final double[] values) {
    this(new DeltaMemoryWriter(values));
  }

  public DeltaSampler(final NDArray values) {
    this(new DeltaMemoryWriter(values));
  }

  @Override
  public void feed(final LogNumber[] data) {
    Random r = Util.R.get();
    assert data.length == length();
    final int dim = length();
    final LogNumber[] v = new LogNumber[data.length];
    for (int i = 0; i < dim; i++) {
      if (r.nextDouble() < this.sampling)
      {
        v[i] = data[i];
      } else {
        v[i] = LogNumber.zero;
      }
    }
    this.values.feed(v);
  }

  public double getSampling() {
    return this.sampling;
  }

  @Override
  public int length() {
    return this.values.length();
  }

  public DeltaSampler setSampling(final double sampling) {
    this.sampling = sampling;
    return this;
  }

}
