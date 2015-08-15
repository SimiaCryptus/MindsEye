package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SigmoidActivationLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  double feedbackAttenuation = 1;
  
  private boolean verbose;
  
  public SigmoidActivationLayer() {
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(inObj.data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double x = input.getData()[i];
      final double f = 1 / (1 + Math.exp(-x));
      final double minDeriv = 0.;
      double d = Math.max(f * (1 - f), minDeriv);
      if (!Double.isFinite(d)) {
        d = minDeriv;
      }
      assert Double.isFinite(d);
      assert minDeriv <= Math.abs(d);
      inputGradient.add(new int[] { i }, 2 * d);
      output.set(i, 2 * f - 1);
    });
    if (isVerbose()) {
      SigmoidActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        NDArray passback = null;
        if (inObj.isAlive()) {
          final NDArray next = new NDArray(data.getDims());
          passback = next;
          IntStream.range(0, next.dim()).forEach(i -> {
            if (Double.isFinite(inputGradient.getData()[i]) && 0 != inputGradient.getData()[i]) {
              next.set(i, data.getData()[i] * inputGradient.getData()[i]);
            }
          });
        }
        if (isVerbose()) {
          SigmoidActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
        }
        inObj.feedback(passback);
      }
      
      @Override
      public boolean isAlive() {
        return true;
      }
    };
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public SigmoidActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
