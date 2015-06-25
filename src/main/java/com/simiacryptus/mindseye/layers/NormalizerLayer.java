package com.simiacryptus.mindseye.layers;

import java.util.stream.DoubleStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class NormalizerLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizerLayer.class);
  
  private final double[] sum;
  private int count = 0;
  private double sumMag = 0;
  
  public NormalizerLayer(int[] outputDims) {
    this.sum = new double[NDArray.dim(outputDims)];
  }
  
  public NNResult eval(final NNResult inObj) {
    count++;
    NDArray translated = 0==count?inObj.data:inObj.data.map((v,i)->{
      sum[i.index] += v;
      double avgi = sum[i.index]/count;
      return v-avgi;
    });
    sumMag += Math.sqrt(DoubleStream.of(translated.data).map(x->x*x).sum());
    double avg = sumMag/count;
    
    NDArray scaled = translated.map((v,i)->{
      return v/avg;
    });
    return new NNResult(scaled) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        if(inObj.isAlive())
        {
          inObj.feedback(data.map((v,i)->v*avg), ctx);
        }
      }
      
      public boolean isAlive() {
        return inObj.isAlive();
      }
    };
  }
  
}
