package com.simiacryptus.mindseye.deltas;

public interface VectorLogic<T extends VectorLogic<T>> {
  
  public T add(T right);
  
  public double dotProduct(T right);
  
  public double l1();
  
  public double l2();
  
  public T scale(double f);
  
  default T unitV() {
    return this.scale(1. / l2());
  };
  
}
