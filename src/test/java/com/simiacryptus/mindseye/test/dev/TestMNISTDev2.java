package com.simiacryptus.mindseye.test.dev;

import org.junit.Test;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;

public class TestMNISTDev2 extends TestMNISTDev {
  public static final class Network extends TestMNISTDev.Network {
    
    public Network() {
      super();
      layers.clear();
      add(new MaxSubsampleLayer(4, 4));

      add(new DenseSynapseLayer(eval(inputSize).data.dim(), eval(inputSize).data.getDims()));
      add(new BiasLayer(eval(inputSize).data.getDims()));
      add(new SigmoidActivationLayer());

//      add(new DenseSynapseLayer(eval(inputSize).data.dim(), eval(inputSize).data.getDims()));
//      add(new BiasLayer(eval(inputSize).data.getDims()));
//      add(new SigmoidActivationLayer());
      
      // layers.add(new DenseSynapseLayer(eval(inputSize).data.dim(), new int[] { 16 })
      // .fillWeights(() -> 0.001 * random.nextGaussian()));
      // layers.add(new SigmoidActivationLayer());
      
      add(new DenseSynapseLayer(eval(inputSize).data.dim(), new int[] { 10 })
          .addWeights(() -> 0.001 * TestMNISTDev2.random.nextGaussian()));
      add(new BiasLayer(eval(inputSize).data.getDims()));
      // layers.add(new BiasLayer(eval(inputSize).data.getDims()));
      add(new SigmoidActivationLayer());
      // add(new SoftmaxActivationLayer());
    }
    
  }
  
  @Test
  public void test() throws Exception {
    super.test();
  }

  @Override
  protected Network getNetwork() {
    Network net = new TestMNISTDev2.Network();
    return net;
  }
}
