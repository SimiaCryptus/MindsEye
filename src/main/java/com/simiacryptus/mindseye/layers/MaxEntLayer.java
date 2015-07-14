package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class MaxEntLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(MaxEntLayer.class);
  double feedbackAttenuation = 1;
  private double factor = -1;
  private boolean reverse = false;
  
  public MaxEntLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(1);
    
    double sum = input.map(x->x*x).sum();
    
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      double sign = Math.signum(input.data[i]);
      final double x = (sign*input.data[i])/(0==sum?1:sum);
      double l = 0==x?0:Math.log(x);
      final double f = factor*x * l;
      double d = (reverse?1:-1)*factor*(1+sign*l);
      assert Double.isFinite(d);
      inputGradient.add(new int[] { i }, d);
      output.add(0, f);
    });
    if (isVerbose()) {
      log.debug(String.format("Feed forward: %s => %s", inObj.data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        NDArray passback = null;
        if (inObj.isAlive()) {
          NDArray next = new NDArray(input.getDims());
          passback = next;
          for(int i=0;i<next.data.length;i++){
            if(Double.isFinite(inputGradient.data[i]) && 0 != inputGradient.data[i]) {
              //double f = output.data[0];
              //f = Math.pow(f, feedbackAttenuation);
              next.set(i, data.data[0] * inputGradient.data[i]);
            }
          }
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

  public MaxEntLayer setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public double getFactor() {
    return factor;
  }

  public MaxEntLayer setFactor(double factor) {
    this.factor = factor;
    return this;
  }

  public boolean isReverse() {
    return reverse;
  }

  public MaxEntLayer setReverse(boolean reverse) {
    this.reverse = reverse;
    return this;
  }

  private boolean verbose;
}
