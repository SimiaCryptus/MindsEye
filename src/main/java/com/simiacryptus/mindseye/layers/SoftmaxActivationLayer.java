package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SoftmaxActivationLayer extends NNLayer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);

  public SoftmaxActivationLayer() {
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    final NDArray exp = inObj.data.map(x -> Math.exp(Math.min(x, 100)));
    final double sum1 = exp.sum();
    final double sum = !Double.isFinite(sum1) || 0. == sum1 ? 1. : sum1;
    final NDArray output = exp.map(x -> x / sum);
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(j -> {
        if (i == j) {
          inputGradient.add(new int[] { i, j }, (1 - exp.data[i]) * exp.data[i]);
        } else {
          inputGradient.add(new int[] { i, j }, -exp.data[i] * exp.data[j]);
        }
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        if (inObj.isAlive()) {
          final double[] delta = data.data;
          inObj.feedback(new NDArray(data.getDims(), org.jblas.Solve.solveLeastSquares(
              new DoubleMatrix(inputGradient.getDims()[0], inputGradient.getDims()[1], inputGradient.data).transpose(),
              new DoubleMatrix(delta.length, 1, delta)).data));
        }
      }

      @Override
      public boolean isAlive() {
        return true;
      }
    };
  }

}
