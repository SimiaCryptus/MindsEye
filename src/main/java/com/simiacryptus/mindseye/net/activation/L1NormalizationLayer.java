package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

public class L1NormalizationLayer extends NNLayer<L1NormalizationLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  private static final long serialVersionUID = -8028442822064680557L;

  public L1NormalizationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final double sum = input.sum();
    final boolean isZeroInput = sum == 0.;
    final NDArray output = input.map(x -> isZeroInput ? x : x / sum);

    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final double[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          final double[] indata = input.getData();
          final NDArray passback = new NDArray(data.getDims());
          double dot = 0;
          for (int i = 0; i < indata.length; i++) {
            dot += delta[i] * indata[i];
          }
          for (int i = 0; i < indata.length; i++) {
            final double d = delta[i];
            passback.set(i, isZeroInput ? d : (d * sum - dot) / (sum * sum));
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
  public List<double[]> state() {
    return Arrays.asList();
  }

}
