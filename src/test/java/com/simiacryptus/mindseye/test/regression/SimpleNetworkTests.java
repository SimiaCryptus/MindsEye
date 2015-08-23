package com.simiacryptus.mindseye.test.regression;

import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.training.PipelineNetwork;

public class SimpleNetworkTests {
  static final Logger log = LoggerFactory.getLogger(SimpleNetworkTests.class);
  
  public static final Random random = new Random();
  
  @Test
  public void test_BasicNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize))
        .add(new BiasLayer(inputSize))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
        .add(new BiasLayer(outSize))
        .trainer(samples)
        .verifyConvergence(100, 0.1, 10);
  }
  
  @Test
  public void test_BasicNN_OR() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        .trainer(samples)
        .verifyConvergence(10, 0.01, 100);
  }

  @Test
  public void test_BasicNN_XOR_3layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 4 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
        .add(new BiasLayer(midSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SigmoidActivationLayer())
        
        .trainer(samples)
        .setMutationAmplitude(10.)
        .verifyConvergence(10, 0.01, 00);
  }

  @Test
  public void test_DualSigmoid() throws Exception {
    final int[] inputSize = new int[] { 1 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { -1 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0 }), new NDArray(outSize, new double[] { .2 }) },
        { new NDArray(inputSize, new double[] { 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).setWeights(new double[] { 1, 1 }).freeze())
        .add(new BiasLayer(midSize).set(new double[] { -1, 1 }))
        .add(new SigmoidActivationLayer())
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).setWeights(new double[] { 1, -1 }).freeze())
        .trainer(samples)
        .setMutationAmount(0)
        .setStaticRate(10.)
        .verifyConvergence(1000, 0.1, 10);
  }

  @Test
  public void test_LinearNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 3 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))
        .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))
        .trainer(samples)
        .setVerbose(false)
        .verifyConvergence(10, 0.1, 100);
  }
  
  @Test
  public void testDenseLinearLayer_2Layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    final double staticRate = 1.;
    
    new PipelineNetwork() //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.5 * SimpleNetworkTests.random.nextGaussian()).freeze())
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)) //
        .trainer(samples)
        .setMutationAmplitude(.5)
        .setStaticRate(staticRate).verifyConvergence(10, 0.01, 10);
    
    new PipelineNetwork() //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * SimpleNetworkTests.random.nextGaussian()).freeze()) //
        .trainer(samples).setStaticRate(staticRate).verifyConvergence(10, 0.01, 10);
    
    new PipelineNetwork() //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)) //
        .trainer(samples).setStaticRate(staticRate).verifyConvergence(10, 0.01, 10);

    new PipelineNetwork() //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)) //
        .trainer(samples).setStaticRate(staticRate).verifyConvergence(10, 0.01, 10);
  }
  
  @Test
  public void testDenseLinearLayer_Basic() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };
    
    new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
        .trainer(samples).setStaticRate(30.).verifyConvergence(10000, 0.01, 10);
  }
  
}
