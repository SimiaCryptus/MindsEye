package com.simiacryptus.mindseye;

import java.util.Arrays;

public class DenseLinearLayer extends NNLayer {
  
  private final int[] outputs;
  private NDArray weights;
  
  public DenseLinearLayer(int inputs, int[] outputs) {
    this.outputs = Arrays.copyOf(outputs, outputs.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputs));
  }
  
  public NNResult eval(NNResult array) {
    NDArray data = null;
    if (1 == 1) throw new RuntimeException("Not Implemented");
    return new NNResult(data) {
      @Override
      public void feedback(NDArray data) {
        throw new RuntimeException("Not Implemented");
      }
    };
  }
  
}
