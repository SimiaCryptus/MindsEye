package com.simiacryptus.mindseye.test.regression;

import java.util.Random;

import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;

public class NetworkElementUnitTests {
  static final Logger log = LoggerFactory.getLogger(NetworkElementUnitTests.class);

  public static final Random random = new Random();

  @Test
  // @Ignore
  public void bias_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { -1, 2 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)), new EntropyLossLayer())//
        .setVerbose(true).verifyConvergence(0.1, 1);
  }

  @Test
  public void bias_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { -1, 1 }) } };
    new Tester()
        .init(samples,
            new DAGNetwork() //
                .add(new BiasLayer(inputSize)),
            new EntropyLossLayer())
        // .setVerbose(true)
        .verifyConvergence(0.01, 100);
  }

  @Test
  // @Ignore
  public void bias_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { -1, 1 }) } };
    new Tester()
        .init(samples, new DAGNetwork().add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()).add(new BiasLayer(inputSize)),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    new Tester()
        .init(samples,
            new DAGNetwork().add(new BiasLayer(inputSize))
                .add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 0 }), new Tensor(outSize, new double[] { 0, -1 }) },
        { new Tensor(inputSize, new double[] { 0, 1 }), new Tensor(outSize, new double[] { 1, 0 }) },
        { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    new Tester()
        .init(samples,
            new DAGNetwork() //
                .add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize)),
            new EntropyLossLayer()) //
        .setVerbose(true) //
        .verifyConvergence(0.1, 1);
  }

  @Test
  public void denseSynapseLayer_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)), new EntropyLossLayer()).verifyConvergence(0.1, 100);
    new Tester()
        .init(samples, new DAGNetwork().add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()).add(new BiasLayer(inputSize)),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
    new Tester()
        .init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
    new Tester().init(samples, new DAGNetwork().add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze())
        .add(new BiasLayer(inputSize)), new EntropyLossLayer()).verifyConvergence(0.1, 100);
    new Tester()
        .init(samples,
            new DAGNetwork().add(new BiasLayer(inputSize))
                .add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };

    new Tester()
        .init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new LinearActivationLayer().addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze()),
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0.5, 2 }), new Tensor(outSize, new double[] { 1, 4 }) } };

    new Tester().init(samples, new DAGNetwork().add(new LinearActivationLayer()), new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void maxSubsampleLayer_feedback() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, -1 }), new Tensor(outSize, new double[] { 1 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new MaxSubsampleLayer(2)), new EntropyLossLayer()).setVerbose(verbose).verifyConvergence(0.1,
        100);
  }

  @Test
  @org.junit.Ignore
  public void n2ActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 4 };
    final int[] outSize = inputSize;
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0, 0, 0 }), new Tensor(outSize, new double[] { 0.2, 0.3, 0.4, 0.1 }) } };
    final DAGNetwork net = new DAGNetwork().add(new BiasLayer(inputSize)).add(new L1NormalizationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void nestingLayer_feedback2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.9, 0.1 }) } };
    final DAGNetwork net = new DAGNetwork() //
        .add(new BiasLayer(inputSize)) //
        .add(new SoftmaxActivationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void sigmoidActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.9, -.9 }) } };
    new Tester().init(samples, new DAGNetwork().add(new BiasLayer(inputSize)).add(new SigmoidActivationLayer()), new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void softmaxActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.1, 0.9 }) } };
    final DAGNetwork net = new DAGNetwork() //
        .add(new BiasLayer(inputSize).setWeights(i -> Util.R.get().nextGaussian()))//
        .add(new SoftmaxActivationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).setVerbose(true) //
        // .setParallel(false)
        .verifyConvergence(0.1, 100);

  }


}
