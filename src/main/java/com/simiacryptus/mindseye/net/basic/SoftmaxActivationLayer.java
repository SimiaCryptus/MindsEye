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
    
    final NDArray exp;
    {
      final DoubleSummaryStatistics summaryStatistics = java.util.stream.DoubleStream.of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
      final double max = summaryStatistics.getMax();
      final double min = summaryStatistics.getMin();
      exp = inObj[0].data.map(x -> {
        return Double.isFinite(x) ? x : min;
      }).map(x -> Math.exp(x - max));
    }
    
    final double sum = exp.sum();
    assert 0. < sum;
    final NDArray output = exp.map(x -> x / sum);
    if (isVerbose()) {
      SoftmaxActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final double[] delta = data.getData();
          final NDArray inputGradient = new NDArray(input.dim(), input.dim());
          final double[] expdata = exp.getData();
          final NDArray passback = new NDArray(data.getDims());
          int dim = expdata.length;
          for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
              double value = 0;
              if (i == j) {
                value = expdata[i] * (sum - expdata[i]) / (sum*sum);
              } else {
                value = -(expdata[i] * expdata[j]) / (sum*sum);
              }
              if (Double.isFinite(value)) {
                passback.add(i, delta[j] * value);
              }
            }
          }
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
