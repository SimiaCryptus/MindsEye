package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class SoftmaxActivationLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
  private boolean verbose;
  
  public SoftmaxActivationLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    double nonlinearity = getNonlinearity();
    final NDArray exp = inObj.data.map(x -> Math.exp(nonlinearity * Math.min(x, 100)));
    final double sum1 = exp.sum();
    final double sum = !Double.isFinite(sum1) || 0. == sum1 ? 1. : sum1;
    final NDArray output = exp.map(x -> x / sum);
    
    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, input.dim()).forEach(j -> {
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
      public void feedback(final LogNDArray data) {
        if (inObj.isAlive()) {
          LogNDArray inputGradientLog = inputGradient.log();
          final LogNumber[] delta = Arrays.copyOf(data.getData(), data.getData().length);
          for (int i = 0; i < delta.length; i++)
            if (delta[i].isNegative()) {
              delta[i] = LogNumber.ZERO;
            }
          
          final LogNDArray passback = new LogNDArray(data.getDims());
          IntStream.range(0, input.dim()).forEach(iinput -> {
            IntStream.range(0, output.dim()).forEach(ioutput -> {
              final LogNumber value = inputGradientLog.get(new int[] { iinput, ioutput });
              if (value.isFinite()) {
                passback.add(iinput, delta[ioutput].multiply(value));
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
        return inObj.isAlive();
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

  protected double getNonlinearity() {
    return 1;
  }
}
