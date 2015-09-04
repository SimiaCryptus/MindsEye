package com.simiacryptus.mindseye.deltas;

public interface VectorLogic<T extends VectorLogic<T>> {
  
  public T scale(double f);
  
  public double dotProduct(T right);
  
  public T add(T right);
  
  public double l2();

  public double l1();
  
}
