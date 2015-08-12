package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;

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
  public void feed(final double[] data) {
    assert data.length == length();
    final int dim = length();
    final double[] v = new double[data.length];
    for (int i = 0; i < dim; i++) {
      if (PipelineNetwork.R.get().nextDouble() < this.sampling)
      {
        v[i] = data[i];
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
