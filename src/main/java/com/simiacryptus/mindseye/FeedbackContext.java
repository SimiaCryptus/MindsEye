package com.simiacryptus.mindseye;

import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;

public class FeedbackContext {

  public double[] invertFeedback(final NDArray gradient, double[] delta) {
    int[] dims = gradient.getDims();
    return org.jblas.Solve.solveLeastSquares(
        new DoubleMatrix(dims[0], dims[1], gradient.data).transpose(),
        new DoubleMatrix(delta.length, 1, delta)).data;
  }

  public void adjust(NNLayer layer, NDArray weightArray, double[] weightDelta) {
    IntStream.range(0, weightArray.dim()).forEach(i->{
      weightArray.add(i, weightDelta[i]);
    });
  }
  
}
