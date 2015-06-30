package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class SigmoidActivationLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  
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
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data) {
        if (inObj.isAlive()) {
          final NDArray next = new NDArray(data.getDims());
          IntStream.range(0, next.dim()).forEach(i -> {
            if((data.data[i]<0)==(inObj.data.data[i]<0))
            {
              next.set(i, data.data[i]);
            } else {
              next.set(i, data.data[i] / inputGradient.data[i]);
            }
          });
          inObj.feedback(next);
        }
      }
      
      @Override
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
