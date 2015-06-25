package com.simiacryptus.mindseye;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.DenseSynapseLayer;

public class TestNetworkUnit {
  public static final Random random = new Random();
  
  
  
  private static final Logger log = LoggerFactory.getLogger(TestNetworkUnit.class);
  
  @Test
  public void testDenseLinearLayer_Basic() throws Exception {
    int[] inputSize = new int[]{2};
    int[] outSize = new int[]{2};
    NDArray[][] samples = new NDArray[][]{
        {new NDArray(inputSize,new double[]{0,1}),new NDArray(outSize,new double[]{1,0})},
        {new NDArray(inputSize,new double[]{1,0}),new NDArray(outSize,new double[]{0,1})}
    };
    DenseSynapseLayer testLayer = new DenseSynapseLayer(NDArray.dim(inputSize), outSize);
    testLayer.fillWeights(()->0.1*Math.random());
    double rms = 0;
    for(int i=0;i<samples.length;i++){
      NDArray input = samples[i][0];
      NDArray output = samples[i][1];
      rms += testLayer.eval(input).errRms(output);
    }
    log.info("RMS Error: {}", rms);
    for (int i = 0; i < 20; i++)
    {
      rms = 0;
      for(int j=0;j<samples.length;j++){
        NDArray input = samples[j][0];
        NDArray output = samples[j][1];
        testLayer.eval(input).learn(.3, output);
        rms += testLayer.eval(input).errRms(output);
      }
      log.info("RMS Error: {}", rms);
    }
    Assert.assertTrue(rms < 0.01);
  }
  
}
