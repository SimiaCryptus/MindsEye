package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

public class NNResult {

  public final NDArray data;

  public NNResult(NDArray data) {
    super();
    this.data = data;
  }
  
  public void feedback(NDArray data) {
  }

  public final void learn(double d, NDArray out) {
    NDArray delta = new NDArray(out.getDims());
    Arrays.parallelSetAll(delta.data, i->(NNResult.this.data.data[i] - out.data[i]) * d);
    feedback(delta);
  }

  public double err(NDArray out) {
    return Math.sqrt(IntStream.range(0, data.dim()).mapToDouble(i->Math.pow(NNResult.this.data.data[i] - out.data[i], 2.)).sum());
  }
  
}
