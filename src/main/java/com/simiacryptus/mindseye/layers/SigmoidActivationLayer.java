package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;

import groovy.lang.Tuple2;

public class SigmoidActivationLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  public static double sigmiod(final double x) {
    return 1 / (1 + Math.exp(-x));
  }

  private boolean balanced = true;
  private boolean verbose;

  public SigmoidActivationLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(inObj[0].data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    final double nonlinearity = getNonlinearity();
    IntStream.range(0, input.dim()).forEach(i -> {

      final double x = input.getData()[i];
      double f = 0. == nonlinearity ? x : SigmoidActivationLayer.sigmiod(x * nonlinearity) / nonlinearity;
      final double minDeriv = 0;
      final double ex = Math.exp(x);
      final double ex1 = 1 + ex;
      double d = 0. == nonlinearity ? 1. : ex / (ex1 * ex1);
      // double d = f * (1 - f);
      if (!Double.isFinite(d) || d < minDeriv) {
        d = minDeriv;
      }
      assert Double.isFinite(d);
      assert minDeriv <= Math.abs(d);
      if (isBalanced()) {
        d = 2 * d;
        f = 2 * f - 1;
      }
      inputGradient.add(new int[] { i }, d);
      output.set(i, f);
    });
    if (isVerbose()) {
      SigmoidActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final LogNDArray inputGradientLog = inputGradient.log();
          final LogNDArray passback = new LogNDArray(data.getDims());
          IntStream.range(0, passback.dim()).forEach(i -> {
            if (inputGradientLog.getData()[i].isFinite()) {
              passback.set(i, data.getData()[i].multiply(inputGradientLog.getData()[i]));
            }
          });
          if (isVerbose()) {
            SigmoidActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          inObj[0].feedback(passback, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  protected double getNonlinearity() {
    return 1;
  }

  public boolean isBalanced() {
    return this.balanced;
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  public SigmoidActivationLayer setBalanced(final boolean balanced) {
    this.balanced = balanced;
    return this;
  }

  public SigmoidActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  public List<Tuple2<Integer, Integer>> permuteOutput(List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  public List<Tuple2<Integer, Integer>> permuteInput(List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }
}
