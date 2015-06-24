package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConvolutionSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);
  
  public final NDArray kernel;
  
  public ConvolutionSynapseLayer(int[] kernelDims, int bandwidth) {
    
    int[] kernelDims2 = Arrays.copyOf(kernelDims, kernelDims.length + 1);
    kernelDims2[kernelDims2.length - 1] = bandwidth;
    this.kernel = new NDArray(kernelDims2);
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    int[] kernelDims = this.kernel.getDims();
    int[] newDims = IntStream.range(0, kernelDims.length).map(
        i -> i == kernelDims.length - 1 ? kernelDims[i] : (inputDims[i] - kernelDims[i] + 1)
        ).toArray();
    final NDArray output = new NDArray(newDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = new NDArray(kernel.dim(), output.dim());
    new NDArray(kernelDims).coordStream().forEach(k -> {
      output.coordStream().forEach(o -> {
        int[] i = IntStream.range(0, k.coords.length).map(idx -> k.coords[idx] + o.coords[idx]).toArray();
        double a = kernel.get(k);
        double b = input.get(i);
        inputGradient.add(new int[] { input.index(i), output.index(o) }, a);
        weightGradient.add(new int[] { kernel.index(k), output.index(o) }, b);
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data) {
        DoubleMatrix weightDelta = org.jblas.Solve.solveLeastSquares(
            new DoubleMatrix(kernel.dim(), output.dim(), weightGradient.data).transpose(),
            new DoubleMatrix(data.dim(), 1, data.data));
        IntStream.range(0, kernel.dim()).forEach(i -> {
          kernel.add(i, weightDelta.data[i]);
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
