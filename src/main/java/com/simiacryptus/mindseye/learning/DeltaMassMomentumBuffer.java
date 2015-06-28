package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.layers.MassParameters;

public class DeltaMassMomentumBuffer implements DeltaBuffer, MassParameters<DeltaMassMomentumBuffer> {
  
  private double decay = 0.9;
  private double mass = 1.;
  private double[] momentum;
  private double[] values;
  
  public DeltaMassMomentumBuffer(final double[] values) {
    this.values = values;
    this.momentum = new double[values.length];
  }
  
  public DeltaMassMomentumBuffer(final NDArray values) {
    this(values.data);
  }
  
  @Override
  public void feed(final double[] data) {
    
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.values[i] += this.momentum[i] = this.decay * this.momentum[i] + data[i] / this.mass;
    }
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
    return this.values.length;
  }
  
  @Override
  public DeltaMassMomentumBuffer setMass(final double mass) {
    this.mass = mass;
    return this;
  }
  
  @Override
  public DeltaMassMomentumBuffer setMomentumDecay(final double momentumDecay) {
    this.decay = momentumDecay;
    return this;
  }

}
