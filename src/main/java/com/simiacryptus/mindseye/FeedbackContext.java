package com.simiacryptus.mindseye;

import java.util.Arrays;
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
    //highpass(weightDelta, 0.4);
    IntStream.range(0, weightArray.dim()).forEach(i->{
      //weightArray.add(i, weightDelta[i] * Math.random());
      weightArray.add(i, weightDelta[i]);
    });
  }

  private void highpass(double[] weightDelta, double percentile) {
    double[] copyDelta = new double[weightDelta.length];
    for(int i=0;i<weightDelta.length;i++)
    {
      copyDelta[i] = Math.abs(weightDelta[i]);
    }
    Arrays.sort(copyDelta);
    double threshold = copyDelta[(int) (copyDelta.length*percentile)];
    for(int i=0;i<weightDelta.length;i++)
    {
      double v = Math.abs(weightDelta[i]);
      if(threshold > v){
        weightDelta[i] = 0;
      }
    }
  }

  private void highpass2(double[] weightDelta) {
    double[] copyDelta = new double[weightDelta.length];
    for(int i=0;i<weightDelta.length;i++)
    {
      copyDelta[i] = Math.abs(weightDelta[i]);
    }
    Arrays.sort(copyDelta);
    for(int i=0;i<weightDelta.length;i++)
    {
      double pct = Arrays.binarySearch(copyDelta, Math.abs(weightDelta[i]))*1./copyDelta.length;
      if(Math.random() < pct){
        weightDelta[i] = 0;
      }
    }
  }

  
}
