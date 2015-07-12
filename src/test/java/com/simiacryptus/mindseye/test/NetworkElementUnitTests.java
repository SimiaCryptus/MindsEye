package com.simiacryptus.mindseye.test;

import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.learning.DeltaInversionBuffer;

public class NetworkElementUnitTests {
  static final Logger log = LoggerFactory.getLogger(NetworkElementUnitTests.class);
  
  public static final Random random = new Random();
  
  @Test
  public void bias_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1, 1 }) }
    };
    new PipelineNetwork()
        .add(new BiasLayer(inputSize))
        .add(new BiasLayer(inputSize).setMass(Double.POSITIVE_INFINITY))
        .trainer(samples).setStaticRate(10.).verifyConvergence(1000, 0.1, 100);
  }
  
  @Test
  public void bias_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1, 1 }) }
    };
    new PipelineNetwork() //
        .add(new BiasLayer(inputSize).setMomentumDecay(0.)) //
        .trainer(samples).setStaticRate(5.).verifyConvergence(1000, 0.01, 100);
  }
  
  @Test
  public void denseSynapseLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    new PipelineNetwork()
        .add(new BiasLayer(inputSize).setMomentumDecay(0.))
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * SimpleNetworkTests.random.nextGaussian()).freeze())
        .trainer(samples).setStaticRate(5.).verifyConvergence(1000, 0.1, 100);
  }
  
  @Test
  public void denseSynapseLayer_train() throws Exception {
    final boolean verbose = false;
    DeltaInversionBuffer.DEBUG = verbose;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, -1 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    for (int i = 0; i < 100; i++) {
      new PipelineNetwork()
          .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)
              .setVerbose(verbose)).trainer(samples).verifyConvergence(10000, 0.1, 1);
    }
  }
  
  @Test
  public void convolutionSynapseLayer_train() throws Exception {
    final boolean verbose = false;
    DeltaInversionBuffer.DEBUG = verbose;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    for (int i = 0; i < 100; i++) {
      new PipelineNetwork() //
          .add(new ConvolutionSynapseLayer(inputSize, 1) //
              .setMomentumDecay(0.) //
              .setVerbose(verbose)) //
          .trainer(samples).setStaticRate(5.).setVerbose(verbose).verifyConvergence(1000, 0.1, 1);
    }
  }
  
  @Test
  public void convolutionSynapseLayer_feedback() throws Exception {
    final boolean verbose = false;
    DeltaInversionBuffer.DEBUG = verbose;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 1 }) }
    };
    for (int i = 0; i < 100; i++) {
      new PipelineNetwork()
          .add(new BiasLayer(inputSize))
          .add(new ConvolutionSynapseLayer(inputSize, 1)
              .addWeights(() -> 10.5 * SimpleNetworkTests.random.nextGaussian())
              .setVerbose(verbose))
          .trainer(samples).setStaticRate(1.).verifyConvergence(1000, 0.1, 1);
    }
  }
  
  @Test
  public void maxSubsampleLayer_feedback() throws Exception {
    final boolean verbose = false;
    DeltaInversionBuffer.DEBUG = verbose;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, -1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    for (int i = 0; i < 100; i++) {
      new PipelineNetwork()
          .add(new BiasLayer(inputSize))
          .add(new MaxSubsampleLayer(2))
          .trainer(samples).setStaticRate(1.).verifyConvergence(1000, 0.1, 1);
    }
  }
  
  @Test
  public void sigmoidActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 1, -1 }) }
    };
    new PipelineNetwork()
        .add(new BiasLayer(inputSize))
        .add(new SigmoidActivationLayer())
        .trainer(samples).setStaticRate(1.).verifyConvergence(1000, 0.1, 100);
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
        .setStaticRate(1.)
        .verifyConvergence(1000, 0.1, 100);
  }
  
}
