package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Coordinate;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;

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
        int[] i = Coordinate.add(k.coords,o.coords);
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
        int dim = kernel.dim();
        for(int i=0;i<dim;i++){
          double[] delta = data.data;
          kernel.add(i, org.jblas.Solve.solveLeastSquares(
          new DoubleMatrix(weightGradient.getDims()[0], weightGradient.getDims()[1], weightGradient.data).transpose(),
          new DoubleMatrix(delta.length, 1, delta)).data[i]);
        }
        if (inObj.isAlive()) {
          double[] delta = data.data;
          inObj.feedback(new NDArray(inputDims, org.jblas.Solve.solveLeastSquares(
          new DoubleMatrix(inputGradient.getDims()[0], inputGradient.getDims()[1], inputGradient.data).transpose(),
          new DoubleMatrix(delta.length, 1, delta)).data));
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }

  public ConvolutionSynapseLayer fillWeights(DoubleSupplier f) {
    Arrays.parallelSetAll(kernel.data, i->f.getAsDouble());
    return this;
  }
  
}
