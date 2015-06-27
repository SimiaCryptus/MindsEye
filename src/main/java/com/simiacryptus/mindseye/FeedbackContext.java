package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.jblas.DoubleMatrix;

@SuppressWarnings("unused")
public class FeedbackContext {
  
  public double[] invertFeedback(final NDArray gradient, double[] delta) {
    int[] dims = gradient.getDims();
    return org.jblas.Solve.solveLeastSquares(
        new DoubleMatrix(dims[0], dims[1], gradient.data).transpose(),
        new DoubleMatrix(delta.length, 1, delta)).data;
  }
  
  public double quantum = 0.;
  
  public void adjust(NNLayer layer, NDArray weightArray, double[] weightDelta) {
    // highpass(weightDelta, 0.4);
    if (quantum > 0.) quantize(weightDelta, quantum);
    int dim = weightArray.dim();
    for(int i=0;i<dim;i++){
      // weightArray.add(i, weightDelta[i] * Math.random());
      weightArray.add(i, weightDelta[i]);
    }
  }
  
  private static void quantize(double[] weightDelta, double quantum) {
    for (int i = 0; i < weightDelta.length; i++)
    {
      double value = weightDelta[i];
      double abs = Math.abs(value);
      if (quantum > abs) {
        if (Math.random() < (abs / quantum)) {
          weightDelta[i] = 0;
        }
        else
        {
          int sign = value < 0 ? -1 : 1;
          weightDelta[i] = (Math.random() * quantum * sign);
        }
      }
    }
  }
  
  private static void quantize2(double[] weightDelta, double quantum) {
    for (int i = 0; i < weightDelta.length; i++)
    {
      double value = weightDelta[i];
      if(0.+value == 0.) continue;
      double abs = Math.abs(value);
      int sign = value < 0 ? -1 : 1;
      int quanta = new PoissonDistribution(abs/quantum,100).sample();
      weightDelta[i] = (quanta * quantum * sign);
    }
  }
  
  private static void highpass(double[] weightDelta, double percentile) {
    double[] copyDelta = new double[weightDelta.length];
    for (int i = 0; i < weightDelta.length; i++)
    {
      copyDelta[i] = Math.abs(weightDelta[i]);
    }
    Arrays.sort(copyDelta);
    double threshold = copyDelta[(int) (copyDelta.length * percentile)];
    for (int i = 0; i < weightDelta.length; i++)
    {
      double v = Math.abs(weightDelta[i]);
      if (threshold > v) {
        weightDelta[i] = 0;
      }
    }
  }
  
  private static void highpass2(double[] weightDelta) {
    double[] copyDelta = new double[weightDelta.length];
    for (int i = 0; i < weightDelta.length; i++)
    {
      copyDelta[i] = Math.abs(weightDelta[i]);
    }
    Arrays.sort(copyDelta);
    for (int i = 0; i < weightDelta.length; i++)
    {
      double pct = Arrays.binarySearch(copyDelta, Math.abs(weightDelta[i])) * 1. / copyDelta.length;
      if (Math.random() < pct) {
        weightDelta[i] = 0;
      }
    }
  }
  
}
