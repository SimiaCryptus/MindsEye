package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

import groovy.lang.Tuple2;

public class SoftmaxActivationLayer extends NNLayer<SoftmaxActivationLayer> {

  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);

  double maxInput = 50;

  public SoftmaxActivationLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    assert 1 < input.dim();
    final DoubleSummaryStatistics summaryStatistics = java.util.stream.DoubleStream.of(inObj[0].data.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
    final double max = summaryStatistics.getMax();
    final NDArray exp = inObj[0].data.map(x -> Double.isFinite(x) ? x : summaryStatistics.getMin()).map(x -> Math.exp(x - max));
    final double sum = exp.sum();
    assert 0. < sum;
    final NDArray output = exp.map(x -> x / sum);
    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    final double[] expdata = exp.getData();

    for (int i = 0; i < expdata.length; i++) {
      for (int j = 0; j < expdata.length; j++) {
        double value = 0;
        if (i == j) {
          value = expdata[i] * (sum - expdata[j]);
        } else {
          value = -(expdata[i] * expdata[j]);
        }
        if (Double.isFinite(value)) {
          inputGradient.add(new int[] { i, j }, value);
        }
      }
    }

    if (isVerbose()) {
      SoftmaxActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final double[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          // for (int i = 0; i < delta.length; i++)
          // if (delta[i].isNegative()) {
          // delta[i] = double.ZERO;
          // }

          final NDArray inputGradientLog = inputGradient;
          final NDArray passback = new NDArray(data.getDims());
          for (int i = 0; i < input.dim(); i++) {
            for (int j = 0; j < output.dim(); j++) {
              final double value = inputGradientLog.get(new int[] { i, j });
              if (Double.isFinite(value)) {
                passback.add(i, delta[j] * value);
              }
            }
            ;
          }
          ;

          if (isVerbose()) {
            SoftmaxActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s; Gradient=%s", output, data, passback, inputGradient));
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

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
