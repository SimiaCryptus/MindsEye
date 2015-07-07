package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;

public class DeltaMassMomentum implements DeltaSink, MassParameters<DeltaMassMomentum> {

  private double decay = 0.;
  private double mass = 1.;
  private double[] momentum;
  private double normalizationFactor = 0;
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
    normalizationFactor += this.decay * normalizationFactor + 1;
    double[] x = new double[v.length];
    for (int i = 0; i < dim; i++) {
      x[i] = v[i] / normalizationFactor;
    }
    this.values.feed(x);
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

  public DeltaMassMomentum setHalflife(final double halflife) {
    return setMomentumDecay(Math.exp(2 * Math.log(0.5) / halflife));
  }
  
}
