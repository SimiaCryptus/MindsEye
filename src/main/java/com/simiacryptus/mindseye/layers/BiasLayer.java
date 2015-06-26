package com.simiacryptus.mindseye.layers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class BiasLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  private final double[] bias;
  
  public BiasLayer(int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
  }
  
  public NNResult eval(final NNResult inObj) {
    NDArray translated = inObj.data.map((v,i)->{
      return v+bias[i.index];
    });
    return new NNResult(translated) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        for(int i=0;i<bias.length;i++)
        {
          bias[i] += data.data[i];
        }
        if(inObj.isAlive())
        {
          inObj.feedback(data, ctx);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
  
}
