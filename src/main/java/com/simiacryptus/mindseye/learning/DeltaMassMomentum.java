package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaMassMomentum implements DeltaSink, MassParameters<DeltaMassMomentum> {

  private double decay = 0.9;
  private double mass = 1.;
  private double[] momentum;
  private DeltaSink values;

  public DeltaMassMomentum() {
    super();
  }

  public DeltaMassMomentum(final double[] values) {
    this(new DeltaMemoryWriter(values));
  }

  public DeltaMassMomentum(final DeltaSink values) {
    this.values = values;
    this.momentum = new double[values.length()];
  }

  public DeltaMassMomentum(final NDArray values) {
    this(values.data);
  }

  @Override
  public void feed(final double[] data) {

    final int dim = length();
    double[] v = new double[data.length];
    for (int i = 0; i < dim; i++) {
      v[i] += this.momentum[i] = this.decay * this.momentum[i] + data[i] / this.mass;
    }
    this.values.feed(v);
  }

  @Override
  public double getMass() {
    return this.mass;
  }

  @Override
  public double getMomentumDecay() {
    return this.decay;
  }

  @Override
  public int length() {
    return this.momentum.length;
  }

  @Override
  public DeltaMassMomentum setMass(final double mass) {
    this.mass = mass;
    return this;
  }

  @Override
  public DeltaMassMomentum setMomentumDecay(final double momentumDecay) {
    this.decay = momentumDecay;
    return this;
  }
  
}
