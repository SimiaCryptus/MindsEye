package com.simiacryptus.mindseye.learning;

public interface MassParameters<T> {
  
  public double getMass();

  public double getMomentumDecay();

  public T setMass(double mass);

  public T setMomentumDecay(double momentumDecay);

}
