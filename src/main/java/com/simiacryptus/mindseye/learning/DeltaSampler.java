package com.simiacryptus.mindseye.learning;

import java.util.Random;

import com.simiacryptus.mindseye.NDArray;

public class DeltaSampler implements DeltaSink {
  public static final Random random = new Random(System.nanoTime());

  private DeltaSink values;
  private double sampling = 2.;

  public DeltaSampler() {
    super();
  }

  public DeltaSampler(final double[] values) {
    this(new DeltaMemoryWriter(values));
  }

  public DeltaSampler(final DeltaSink values) {
    this.values = values;
  }

  public DeltaSampler(final NDArray values) {
    this(new DeltaMemoryWriter(values));
  }

  @Override
  public void feed(final double[] data) {
    assert(data.length==length());
    final int dim = length();
    double[] v = new double[data.length];
    for (int i = 0; i < dim; i++) {
      if(random.nextDouble() < sampling) 
      {
        v[i] = data[i];
      }
    }
    this.values.feed(v);
  }

  @Override
  public int length() {
    return values.length();
  }

  public double getSampling() {
    return sampling;
  }

  public DeltaSampler setSampling(double sampling) {
    this.sampling = sampling;
    return this;
  }
  
}
