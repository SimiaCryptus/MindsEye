package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class SoftmaxActivationLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
  public SoftmaxActivationLayer() {
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray inputGradient = new NDArray(input.dim(), input.dim());
    NDArray exp = inObj.data.map(x->Math.exp(Math.min(x, 100)));
    double sum1 = exp.sum();
    double sum = (!Double.isFinite(sum1) || 0. == sum1)?1.:sum1;
    NDArray output = exp.map(x->x/sum);
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(j -> {
        if(i==j) {
          inputGradient.add(new int[]{i,j}, (sum-exp.data[i])*exp.data[i] / sum*sum);
        }
        else
        {
          inputGradient.add(new int[]{i,j}, exp.data[j]*exp.data[i] / sum*sum);
        }
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        if (inObj.isAlive()) {
          inObj.feedback(new NDArray(data.getDims(), ctx.invertFeedback(inputGradient, data.data)), ctx);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
