package com.simiacryptus.mindseye.test.regression;

import java.util.Random;

import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;

public class NetworkElementUnitTests {
  static final Logger log = LoggerFactory.getLogger(NetworkElementUnitTests.class);

  public static final Random random = new Random();

  @Test
  public void _denseSynapseLayer_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };

    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
  }

  @Test
  // @Ignore
  public void bias_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1, 2 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    //.add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .trainer(samples).verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void bias_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1, 1 }) }
    };
    new PipelineNetwork() //
    .add(new BiasLayer(inputSize)) 
    .trainer(samples) //
    //.setVerbose(true)
    .verifyConvergence(0, 0.01, 10);
  }

  @Test
  // @Ignore
  public void bias_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1, 1 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .add(new BiasLayer(inputSize))
    .trainer(samples).verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void convolutionSynapseLayer_feedback() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new ConvolutionSynapseLayer(inputSize, 1)
    .addWeights(() -> 10.5 * SimpleNetworkTests.random.nextGaussian())
    .setVerbose(verbose)
    .freeze())
    .trainer(samples)
    .setVerbose(verbose)
    .setStaticRate(.1)
    .verifyConvergence(0, 0.1, 10);
  }

  @Test
  @Ignore
  public void convolutionSynapseLayer_train() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork() //
    .add(new ConvolutionSynapseLayer(inputSize, 1).setVerbose(verbose)) //
    .trainer(samples) //
    // .setStaticRate(.5)
    .setVerbose(verbose) //
    // .setMinDynamicRate(0).setMaxDynamicRate(0.5)
    .verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void convolutionSynapseLayer_train2() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
    .add(new ConvolutionSynapseLayer(inputSize, 1)
    .addWeights(() -> 10.5 * SimpleNetworkTests.random.nextGaussian())
    .setVerbose(verbose)
    .freeze())
    .add(new BiasLayer(outSize))
    .trainer(samples)
    .setVerbose(verbose)
    .setStaticRate(.1)
    .verifyConvergence(0, 0.1, 10);
  }

  @Test
  public void denseSynapseLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void denseSynapseLayer_train() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, -1 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    new PipelineNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).setVerbose(verbose)) //
    .trainer(samples) //
    // .setStaticRate(.25).setMutationAmount(1)
    .setVerbose(verbose)
    .verifyConvergence(0, 0.1, 1);
  }

  @Test
  public void denseSynapseLayer_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
    new PipelineNetwork()
    .add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .add(new BiasLayer(inputSize))
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .add(new BiasLayer(inputSize))
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 100);
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .trainer(samples)
    // .setStaticRate(.01)
    // .setVerbose(true)
    .verifyConvergence(0, 0.1, 10);
  }

  @Test
  public void maxSubsampleLayer_feedback() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, -1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new MaxSubsampleLayer(2))
    .trainer(samples)
    .setVerbose(verbose)
    .verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void sigmoidActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.9, -.9 }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new SigmoidActivationLayer())
    .trainer(samples)
    .verifyConvergence(0, 0.1, 100);
  }

  @Test
  public void softmaxActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0., 1. }) }
    };
    new PipelineNetwork()
    .add(new BiasLayer(inputSize))
    .add(new SoftmaxActivationLayer())
    .trainer(samples)
    .verifyConvergence(0, 0.1, 100);
  }

}
