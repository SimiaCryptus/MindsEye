package com.simiacryptus.mindseye.layers;

public interface MassParameters<T> {
  public double getMomentumDecay();
  public T setMomentumDecay(double momentumDecay);
  public double getMass();
  public T setMass(double mass);
}
