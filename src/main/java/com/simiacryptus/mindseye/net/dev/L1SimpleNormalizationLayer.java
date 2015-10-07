package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaSet;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;

import groovy.lang.Tuple2;

public class L1SimpleNormalizationLayer extends NNLayer<L1SimpleNormalizationLayer> {

  private static final long serialVersionUID = -8028442822064680557L;
  private static final Logger log = LoggerFactory.getLogger(L1SimpleNormalizationLayer.class);

  public L1SimpleNormalizationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final double s1 = input.sum();
    final double sum = s1 == 0. ? 1 : s1;
    final NDArray output = input.map(x -> x / sum);

    if (isVerbose()) {
      L1SimpleNormalizationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final double[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          final NDArray passback = new NDArray(data.getDims());
          for (int i = 0; i < input.dim(); i++) {
            passback.add(i, delta[i] / sum);
          }
          ;
          if (isVerbose()) {
            L1SimpleNormalizationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
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
