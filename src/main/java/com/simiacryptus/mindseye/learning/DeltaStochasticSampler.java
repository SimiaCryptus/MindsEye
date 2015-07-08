package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaStochasticSampler implements DeltaSink {

  private DeltaSink values;
  private double sampling = 1;

  public DeltaStochasticSampler() {
    super();
  }

  public DeltaStochasticSampler(final double[] values) {
    this(new DeltaMemoryWriter(values));
  }

  public DeltaStochasticSampler(final DeltaSink values) {
    this.values = values;
  }

  public DeltaStochasticSampler(final NDArray values) {
    this(new DeltaMemoryWriter(values));
  }

  @Override
  public void feed(final double[] data) {
    assert(data.length==length());
    final int dim = length();
    double[] v = new double[data.length];
    for (int i = 0; i < dim; i++) {
      if(Math.random() < sampling) 
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

  public DeltaStochasticSampler setSampling(double sampling) {
    this.sampling = sampling;
    return this;
  }
  
}
