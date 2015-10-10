package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

import groovy.lang.Tuple2;

public class L1NormalizationLayer extends NNLayer<L1NormalizationLayer> {

  private static final long serialVersionUID = -8028442822064680557L;
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);

  public L1NormalizationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final double sum = input.sum();
    boolean isZeroInput = sum == 0.;
    final NDArray output = input.map(x -> isZeroInput ? x : (x / sum));

    if (isVerbose()) {
      L1NormalizationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final double[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          double[] indata = input.getData();
          final NDArray passback = new NDArray(data.getDims());
          double dot = 0;
          for (int i = 0; i < indata.length; i++) {
            dot += delta[i] * indata[i];
          }
          for (int i = 0; i < indata.length; i++) {
            double d = delta[i];
            passback.set(i, isZeroInput?d:((d*sum-dot)/(sum*sum)));
          }
          if (isVerbose()) {
            L1NormalizationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
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
