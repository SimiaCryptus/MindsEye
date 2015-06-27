package com.simiacryptus.mindseye;

import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;

public class TestNetworkUnit {
  public static final Random random = new Random();
  
  static final Logger log = LoggerFactory.getLogger(TestNetworkUnit.class);
  
  @Test
  public void testDenseLinearLayer_Basic() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] outSize = new int[] { 2 };
    NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 1000, 0.01, 100);
  }
  
  @Test
  public void testDenseLinearLayer_2Layer() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] outSize = new int[] { 2 };
    NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
    
    new PipelineNetwork() 
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * random.nextGaussian()).freeze())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()).freeze())
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
  }
  
  @Test
  public void test_LinearNN() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .setRate(0.1)
        .test(samples, 1000, 0.01, 100);
  }
  
  @Test
  public void test_BasicNN() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(inputSize))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .setRate(0.01)
        .test(samples, 100000, 0.01, 10);
  }
  
  @Test
  public void test_BasicNN_XOR() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] midSize = new int[] { 2 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork() {
//      
//      @Override
//      protected DenseSynapseLayer mutate(DenseSynapseLayer l) {
//        //l.addWeights(() -> 0.05 * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
//        //return super.mutate(l);
//        return l;
//      }
//      
//      @Override
//      public double getRate(int iteration) {
//        return 0.005;
//      }
      
    }
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
//        .setRate(0.0001).setQuantum(0.)
        .setVerbose(true)
        .test(samples, 100000, 0.01, 1);
  }
  
  @Test
  public void test_BasicNN_OR() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] midSize = new int[] { 2 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork() 
        // Becomes unstable if these are added:
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        // Works okay:
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        .test(samples, 100000, 0.01, 10);
  }
  
  @Test
  public void test_BasicNN_AND() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] midSize = new int[] { 2 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork() 
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        .test(samples, 100000, 0.01, 10);
  }
  
}
