package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
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
      double enx = Math.exp(-x);
      double enx1 = 1 + enx;
      double minDeriv = 0.0000001;
      double d = Math.max(enx / enx1*enx1, minDeriv);
      if(!Double.isFinite(d)) d = minDeriv;
      assert(Double.isFinite(d));
      assert(0.0001<Math.abs(d));
      double f = 1./enx1;
      inputGradient.add(new int[] { i }, 2*d);
      output.set(i, 2*(f-0.5));
      //output.set(i, f);
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        if (inObj.isAlive()) {
          NDArray next = new NDArray(data.getDims());
          IntStream.range(0, next.dim()).forEach(i -> {
            next.set(i, data.data[i] / inputGradient.data[i]);
          });
          inObj.feedback(next, ctx);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
