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

public class ExpActivationLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(ExpActivationLayer.class);

  private boolean verbose;

  public ExpActivationLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    assert 1 == inObj.length;
    final NNResult in = inObj[0];
    final NDArray input = in.data;
    final NDArray output = new NDArray(in.data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double x = input.getData()[i];
      final double max = 700;// Math.log(Double.MAX_VALUE);
      final double bounded = Math.max(Math.min(max, x), -max);
      final double ex = Math.exp(bounded);
      final double d = ex;
      final double f = ex;
      inputGradient.set(new int[] { i }, d);
      output.set(i, f);
    });
    if (isVerbose()) {
      ExpActivationLayer.log.debug(String.format("Feed forward: %s => %s", in.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (in.isAlive()) {
          final LogNDArray passback = new LogNDArray(data.getDims());
          IntStream.range(0, passback.dim()).forEach(i -> {
            final double x = data.getData()[i];
            final double dx = inputGradient.getData()[i];
            passback.set(i, x*(dx));
          });
          if (isVerbose()) {
            ExpActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          in.feedback(passback, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return in.isAlive();
      }
    };
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  public ExpActivationLayer setVerbose(final boolean verbose) {
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
