package com.simiacryptus.mindseye.layers;

import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.LogNDArray;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class MaxEntLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(MaxEntLayer.class);
  private double factor = -1;
  private boolean reverse = false;
  
  private boolean verbose;
  
  public MaxEntLayer() {
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(1);
    
    final double sum = input.map(x -> Math.abs(x)).sum();
    
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double sign = Math.signum(input.getData()[i]);
      final double x = sign * input.getData()[i] / (0 == sum ? 1 : sum);
      final double l = 0 == x ? 0 : Math.log(x);
      final double f = this.factor * x * l;
      final double d = (this.reverse ? 1 : -1) * this.factor * (1 + sign * l);
      assert Double.isFinite(d);
      inputGradient.add(new int[] { i }, d);
      output.add(0, f);
    });
    if (isVerbose()) {
      MaxEntLayer.log.debug(String.format("Feed forward: %s => %s", inObj.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data) {
        if (inObj.isAlive()) {
          LogNDArray inputGradientLog = inputGradient.log();
          LogNDArray passback = new LogNDArray(input.getDims());
          for (int i = 0; i < passback.getData().length; i++) {
            if (Double.isFinite(inputGradientLog.getData()[i])) {
              // double f = output.data[0];
              // f = Math.pow(f, feedbackAttenuation);
              passback.set(i, inputGradientLog.getData()[i]);
            }
          }
          if (isVerbose()) {
            MaxEntLayer.log.debug(String.format("Feed back @ %s: => %s", output, passback));
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

  public double getFactor() {
    return this.factor;
  }

  public boolean isReverse() {
    return this.reverse;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public MaxEntLayer setFactor(final double factor) {
    this.factor = factor;
    return this;
  }

  public MaxEntLayer setReverse(final boolean reverse) {
    this.reverse = reverse;
    return this;
  }

  public MaxEntLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
