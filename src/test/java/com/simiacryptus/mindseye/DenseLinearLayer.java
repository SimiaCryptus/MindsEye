package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DenseLinearLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseLinearLayer.class);
  
  private final int[] outputDims;
  public final NDArray weights;
  
  public DenseLinearLayer(int inputs, int[] outputDims) {
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
//        if(0. != b)
//        {
//          log.debug("yay");
//        }
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data) {
        DoubleMatrix inputDelta = org.jblas.Solve.solveLeastSquares(
            new DoubleMatrix(output.dim(), input.dim(), inputGradient.data), 
            new DoubleMatrix(1, data.dim(), data.data));
        DoubleMatrix weightDelta = org.jblas.Solve.solveLeastSquares(
            new DoubleMatrix(output.dim(), weights.dim(), weightGradient.data), 
            new DoubleMatrix(1, data.dim(), data.data));
        IntStream.range(0, weights.dim()).forEach(i->{
          weights.add(i, weightDelta.data[i]);
        });
        inObj.feedback(new NDArray(inputDims, inputDelta.data));
      }
    };
  }
  
}
