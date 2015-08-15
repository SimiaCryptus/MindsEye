package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SoftmaxActivationLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
  private boolean verbose;
  
  public SoftmaxActivationLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray exp = inObj.data.map(x -> Math.exp(Math.min(x, 100)));
    final double sum1 = exp.sum();
    final double sum = !Double.isFinite(sum1) || 0. == sum1 ? 1. : sum1;
    final NDArray output = exp.map(x -> x / sum);
    
    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(j -> {
        double value = 0;
        if (i == j) {
          value = exp.getData()[i] * (sum - exp.getData()[i]);
        } else {
          value = -(exp.getData()[i] * exp.getData()[j]);
        }
        if (Double.isFinite(value)) {
          inputGradient.add(new int[] { i, j }, value);
        }
      });
    });
    
    if (isVerbose()) {
      SoftmaxActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        if (inObj.isAlive()) {
          final double[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          for (int i = 0; i < delta.length; i++)
            if (delta[i] < 0) {
              delta[i] = 0;
            }
          
          final NDArray passback = new NDArray(data.getDims());
          IntStream.range(0, input.dim()).forEach(iinput -> {
            IntStream.range(0, output.dim()).forEach(ioutput -> {
              final double value = inputGradient.get(new int[] { iinput, ioutput });
              if (Double.isFinite(value) && 0.00001 < Math.abs(value)) {
                passback.add(iinput, delta[ioutput] / value);
              }
            });
          });
          
          if (isVerbose()) {
            SoftmaxActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s; Gradient=%s", output, data, passback, inputGradient));
          }
          inObj.feedback(passback);
        }
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
  
  public SoftmaxActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
