package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

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
      double f = 1 / (1 + Math.exp(-x));
      double minDeriv = 0.000001;
      double d = Math.max(f*(1-f), minDeriv);
      if(!Double.isFinite(d)) d = minDeriv;
      assert(Double.isFinite(d));
      assert(minDeriv<=Math.abs(d));
      inputGradient.add(new int[] { i }, 2*d);
      output.set(i, 2*f-1);
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