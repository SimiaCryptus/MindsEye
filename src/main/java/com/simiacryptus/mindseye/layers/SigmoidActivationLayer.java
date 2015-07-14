package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SigmoidActivationLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  double feedbackAttenuation = 0;
  
  public SigmoidActivationLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(inObj.data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double x = input.data[i];
      final double f = 1 / (1 + Math.exp(-x));
      final double minDeriv = 0.000001;
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
      log.debug(String.format("Feed forward: %s => %s", inObj.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        NDArray passback = null;
        if (inObj.isAlive()) {
          NDArray next = new NDArray(data.getDims());
          passback = next;
          IntStream.range(0, next.dim()).forEach(i -> {
            if(Double.isFinite(inputGradient.data[i]) && 0 != inputGradient.data[i]) {
              double f = (output.data[i]<0==data.data[i]<0)?(1-Math.abs(output.data[i])):1;
              f = Math.pow(f, feedbackAttenuation);
              next.set(i, f * data.data[i] * inputGradient.data[i]);
            }
          });
        }
        if (isVerbose()) {
          log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
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
    return verbose;
  }

  public SigmoidActivationLayer setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  private boolean verbose;
}
