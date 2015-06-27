package com.simiacryptus.mindseye.learning;

import com.simiacryptus.mindseye.NDArray;


public class DeltaMomentumBuffer implements DeltaBuffer {

  private NDArray values;
  private NDArray momentum;
  private double decay = 0.9;

  public DeltaMomentumBuffer(NDArray values) {
    this.values = values;
    this.momentum = new NDArray(values.getDims());
  }

  @Override
  public int length() {
    return values.dim();
  }

  @Override
  public void feed(double[] data) {

    int dim = length();
    for(int i=0;i<dim;i++){
      values.data[i] += (momentum.data[i] = decay * (momentum.data[i]+data[i]));
    }
  }

  public double getDecay() {
    return decay;
  }

  public DeltaMomentumBuffer setDecay(double decay) {
    this.decay = decay;
    return this;
  }
  
}
