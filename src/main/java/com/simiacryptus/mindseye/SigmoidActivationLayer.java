package com.simiacryptus.mindseye;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SigmoidActivationLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  
  public SigmoidActivationLayer() {
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(inObj.data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      double x = input.data[i];
      double enx = Math.exp(-x);
      double enx1 = 1 + enx;
      double d = Math.max(enx / enx1*enx1, 0.001);
      if(!Double.isFinite(d)) d = 0.001;
      assert(Double.isFinite(d));
      assert(0.0001<Math.abs(d));
      double f = 1./enx1;
      inputGradient.add(new int[] { i }, d);
      output.set(i, f);
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data) {
        if (inObj.isAlive()) {
          NDArray next = new NDArray(data.getDims());
          IntStream.range(0, next.dim()).forEach(i -> {
            next.set(i, data.data[i] / inputGradient.data[i]);
          });
          inObj.feedback(next);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
