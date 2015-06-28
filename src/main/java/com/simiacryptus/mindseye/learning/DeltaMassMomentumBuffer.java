package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MassParameters;


public class DeltaMassMomentumBuffer implements DeltaBuffer, MassParameters<DeltaMassMomentumBuffer> {

  private double[] values;
  private double[] momentum;
  private double decay = 0.9;
  private double mass = 1.;

  public DeltaMassMomentumBuffer(double[] values) {
    this.values = values;
    this.momentum = new double[values.length];
  }

  public DeltaMassMomentumBuffer(NDArray values) {
    this(values.data);
  }

  @Override
  public int length() {
    return values.length;
  }

  @Override
  public void feed(double[] data) {

    int dim = length();
    for(int i=0;i<dim;i++){
      values[i] += (momentum[i] = decay * (momentum[i]+data[i]/mass));
    }
  }

  public double getMass() {
    return mass;
  }

  public DeltaMassMomentumBuffer setMass(double mass) {
    this.mass = mass;
    return this;
  }

  @Override
  public double getMomentumDecay() {
    return decay;
  }

  @Override
  public DeltaMassMomentumBuffer setMomentumDecay(double momentumDecay) {
    decay=momentumDecay;
    return this;
  }
  
}
