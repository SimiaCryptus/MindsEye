package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DenseSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  private final int[] outputDims;
  public final NDArray weights;
  
  public DenseSynapseLayer(int inputs, int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final NDArray output = new NDArray(outputDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = new NDArray(weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        double a = weights.get(i, o);
        double b = input.data[i];
        inputGradient.add(new int[] { i, o }, a);
        weightGradient.add(new int[] { weights.index(i, o), o }, b);
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data) {
        DoubleMatrix weightDelta = org.jblas.Solve.solveLeastSquares(
            new DoubleMatrix(weights.dim(), output.dim(), weightGradient.data).transpose(), 
            new DoubleMatrix(data.dim(), 1, data.data));
        IntStream.range(0, weights.dim()).forEach(i->{
          weights.add(i, weightDelta.data[i]);
        });

        if (inObj.isAlive()) {
          DoubleMatrix inputDelta = org.jblas.Solve.solveLeastSquares(
              new DoubleMatrix(input.dim(), output.dim(), inputGradient.data).transpose(),
              new DoubleMatrix(data.dim(), 1, data.data));
          inObj.feedback(new NDArray(inputDims, inputDelta.data));
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
