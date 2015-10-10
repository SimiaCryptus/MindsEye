package com.simiacryptus.mindseye.test.regression;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dev.LinearActivationLayer;
import com.simiacryptus.mindseye.net.dev.SynapseActivationLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;

import groovy.lang.Tuple2;

public class NetworkElementUnitTests {
  static final Logger log = LoggerFactory.getLogger(NetworkElementUnitTests.class);

  public static final Random random = new Random();

  @Test
  // @Ignore
  public void bias_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1, 2 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize).setVerbose(true)), (NNLayer<?>) new EntropyLossLayer())//
        .setVerbose(true)
        .verifyConvergence(0.1, 1);
  }

  @Test
  public void bias_permute_back() throws Exception {
    final BiasLayer layer = new BiasLayer(new int[] { 5 });
    java.util.Arrays.setAll(layer.bias, i -> i);
    layer.permuteOutput(java.util.Arrays.asList(new Tuple2<>(1, 3), new Tuple2<>(2, 1), new Tuple2<>(3, 2)));
    Assert.assertArrayEquals(layer.bias, new double[] { 0, 2, 3, 1, 4 }, 0.001);
  }

  @Test
  public void bias_permute_fwd() throws Exception {
    final BiasLayer layer = new BiasLayer(new int[] { 5 });
    java.util.Arrays.setAll(layer.bias, i -> i);
    layer.permuteInput(java.util.Arrays.asList(new Tuple2<>(1, 3), new Tuple2<>(2, 1), new Tuple2<>(3, 2)));
    Assert.assertArrayEquals(layer.bias, new double[] { 0, 2, 3, 1, 4 }, 0.001);
  }

  @Test
  public void bias_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1, 1 }) } };
    new Tester().init(samples, new DAGNetwork() //
    .add(new BiasLayer(inputSize)), (NNLayer<?>) new EntropyLossLayer())
        // .setVerbose(true)
        .verifyConvergence(0.01, 100);
  }

  @Test
  // @Ignore
  public void bias_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1, 1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()).add(new BiasLayer(inputSize)), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_feedback2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };

    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_train() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, -1 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };
    new Tester().init(samples, new DAGNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).setVerbose(verbose)), (NNLayer<?>) new EntropyLossLayer()) //
        .setVerbose(true) //
        .verifyConvergence(0.1, 1);
  }

  @Test
  public void denseSynapseLayer_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()).add(new BiasLayer(inputSize)), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
    new Tester().init(samples, new DAGNetwork()
    .add(new BiasLayer(inputSize))
    .add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer())
      .verifyConvergence(0.1, 100);
    new Tester().init(samples, new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
    .add(new BiasLayer(inputSize)), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };

    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new LinearActivationLayer().addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0.5, 2 }), new NDArray(outSize, new double[] { 1, 4 }) } };

    new Tester().init(samples, new DAGNetwork().add(new LinearActivationLayer()), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void maxSubsampleLayer_feedback() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, -1 }), new NDArray(outSize, new double[] { 1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new MaxSubsampleLayer(2)), (NNLayer<?>) new EntropyLossLayer()).setVerbose(verbose).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void n2ActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 4 };
    final int[] outSize = inputSize;
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0, 0, 0 }), new NDArray(outSize, new double[] { 0.2, 0.3, 0.4, 0.1 }) } };
    DAGNetwork net = new DAGNetwork().add(new BiasLayer(inputSize)).add(new L1NormalizationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void nestingLayer_feedback2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.9, 0.1 }) } };
    DAGNetwork net = new DAGNetwork() //
    .add(new BiasLayer(inputSize)) //
    .add(new SoftmaxActivationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void sigmoidActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.9, -.9 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new SigmoidActivationLayer()), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void softmaxActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.1, 0.9 }) } };
    boolean verbose = true;
    DAGNetwork net = new DAGNetwork() //
    .add(new BiasLayer(inputSize).setWeights(i->Util.R.get().nextGaussian()).setVerbose(verbose))//
    .add(new SoftmaxActivationLayer().setVerbose(verbose));
    new Tester().init(samples, net, new EntropyLossLayer()).setVerbose(true) //
        //.setParallel(false)
        .verifyConvergence(0.1, 100);

  }

  @Test
  public void synapseActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1, -1 }) } };

    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new SynapseActivationLayer(NDArray.dim(inputSize)).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void synapseActivationLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0.5, 2 }), new NDArray(outSize, new double[] { 1, -1 }) } };

    new Tester().init(samples, new DAGNetwork().add(new SynapseActivationLayer(NDArray.dim(inputSize))), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

}
