package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.jblas.DoubleMatrix;

@SuppressWarnings("unused")
public class FeedbackContext {
  
  public double[] invertFeedback(final NDArray gradient, double[] delta) {
    double[] mcdelta = Arrays.copyOf(delta, delta.length);
    for(int i=0;i<mcdelta.length;i++) mcdelta[i] *= Math.random()<0.5?1:1;
    int[] dims = gradient.getDims();
    return org.jblas.Solve.solveLeastSquares(
        new DoubleMatrix(dims[0], dims[1], gradient.data).transpose(),
        new DoubleMatrix(delta.length, 1, mcdelta)).data;
  }
  
  public void adjust(NNLayer layer, NDArray weightArray, double[] weightDelta) {
    int dim = weightArray.dim();
    for(int i=0;i<dim;i++){
      weightArray.add(i, weightDelta[i]);
    }
  }
  
  
}
