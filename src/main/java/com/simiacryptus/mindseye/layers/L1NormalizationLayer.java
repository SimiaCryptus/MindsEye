package com.simiacryptus.mindseye.layers;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;

// XXX: Actually, L1
public class L1NormalizationLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);

  private boolean verbose;

  public L1NormalizationLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final double s1 = input.sum();
    final double sum = s1 == 0. ? 1 : s1;
    final NDArray output = input.map(x -> x / sum);

    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    final double[] indata = input.getData();
    for (int i = 0; i < indata.length; i++) {
      for (int j = 0; j < indata.length; j++) {
        double value = 0;
        if (i == j) {
          // XXX: Are i and j are reversed here?
          value = (sum - indata[i]) / (sum * sum);
        } else {
          value = -indata[j] / (sum * sum);
        }
        if (Double.isFinite(value)) {
          inputGradient.add(new int[] { i, j }, value);
        }
      }
      ;
    }
    ;

    if (isVerbose()) {
      L1NormalizationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final LogNumber[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          final LogNDArray inputGradientLog = inputGradient.log();
          final LogNDArray passback = new LogNDArray(data.getDims());
          for (int i = 0; i < input.dim(); i++) {
            for (int j = 0; j < output.dim(); j++) {
              passback.add(i, delta[j].multiply(inputGradientLog.get(new int[] { i, j })));
            }
            ;
          }
          ;
          if (isVerbose()) {
            L1NormalizationLayer.log.debug(String.format("Feed back @ %s: %s => %s; Gradient=%s", output, data, passback, inputGradient));
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
  public boolean isVerbose() {
    return this.verbose;
  }

  public L1NormalizationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
