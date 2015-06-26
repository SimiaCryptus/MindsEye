package com.simiacryptus.mindseye;

import java.util.Random;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.junit.Assert;
import org.junit.Ignore;
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
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
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
    
    new PipelineNetwork() {
      
      @Override
      protected void mutate() {
        super.mutate();
      }
    }
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
    
    new PipelineNetwork() {
      
      @Override
      protected void mutate() {
        super.mutate();
      }
    }
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).fillWeights(() -> 0.1 * random.nextGaussian()).freeze())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .setRate(0.1)
        .test(samples, 10000, 0.01, 100);
    
    new PipelineNetwork() {
      
      @Override
      protected void mutate() {
        super.mutate();
      }
    }
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()).freeze())
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
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
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
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(inputSize))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        .setRate(0.01)
        .test(samples, 100000, 0.01, 100);
  }
  
  @Test
  public void test_BasicNN_XOR() throws Exception {
    int[] inputSize = new int[] { 2 };
    int[] midSize = new int[] { 12 };
    int[] outSize = new int[] { 1 };
    NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork() {
      
      @Override
      public double getRate(int iteration) {
        return super.getRate();
      }
      
    }
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).fillWeights(() -> 0.1 * random.nextGaussian()))
        .add(new BiasLayer(outSize))
        // .setRate(0.0001).setQuantum(0.0001)
        .test(samples, 100000, 0.01, 100);
  }
  
}
