package com.simiacryptus.mindseye.test.regression;

import java.util.Random;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.PipelineNetwork;
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
    PipelineNetwork pipelineNetwork = new PipelineNetwork();
    pipelineNetwork.add(new BiasLayer(inputSize));
    new Tester().init(samples, pipelineNetwork, new EntropyLossLayer())//
        .setVerbose(true).verifyConvergence(0.1, 1);
  }

  @Test
  public void bias_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { -1, 1 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize));
    new Tester().init(samples, network, new EntropyLossLayer())
        // .setVerbose(true)
        .verifyConvergence(0.01, 100);
  }

  @Test
  // @Ignore
  public void bias_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { -1, 1 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    network.add(new BiasLayer(inputSize));
    new Tester().init(samples, network,new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize));
    network.add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    new Tester().init(samples,network,new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void denseSynapseLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 0 }), new Tensor(outSize, new double[] { 0, -1 }) },
        { new Tensor(inputSize, new double[] { 0, 1 }), new Tensor(outSize, new double[] { 1, 0 }) },
        { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize));
    new Tester().init(samples, network, new EntropyLossLayer())
        .setVerbose(true) //
        .verifyConvergence(0.1, 1);
  }

  @Test
  public void denseSynapseLayer_train2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };
    PipelineNetwork network1 = new PipelineNetwork();
    network1.add(new BiasLayer(inputSize));
    new Tester().init(samples, network1, new EntropyLossLayer()).verifyConvergence(0.1, 100);
    PipelineNetwork network3 = new PipelineNetwork();
    network3.add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    network3.add(new BiasLayer(inputSize));
    new Tester()
        .init(samples, network3,
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
    PipelineNetwork network5 = new PipelineNetwork();
    network5.add(new BiasLayer(inputSize));
    network5.add(new BiasLayer(inputSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    new Tester()
        .init(samples, network5,
            new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
    PipelineNetwork network2 = new PipelineNetwork();
    network2.add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    network2.add(new BiasLayer(inputSize));
    new Tester().init(samples, network2, new EntropyLossLayer()).verifyConvergence(0.1, 100);
    PipelineNetwork network6 = new PipelineNetwork();
    network6.add(new BiasLayer(inputSize));
    network6.add(new DenseSynapseLayer(Tensor.dim(inputSize), outSize).addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    new Tester().init(samples,network6,new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 1, 1 }), new Tensor(outSize, new double[] { 1, -1 }) } };

    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize));
    network.add(new LinearActivationLayer().addWeights(() -> 10 * SimpleNetworkTests.random.nextGaussian()).freeze());
    new Tester().init(samples, network,new EntropyLossLayer())
        .verifyConvergence(0.1, 100);
  }

  @Test
  public void linearActivationLayer_train() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0.5, 2 }), new Tensor(outSize, new double[] { 1, 4 }) } };

    PipelineNetwork network = new PipelineNetwork();
    network.add(new LinearActivationLayer());
    new Tester().init(samples, network, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void maxSubsampleLayer_feedback() throws Exception {
    final boolean verbose = false;
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, -1 }), new Tensor(outSize, new double[] { 1 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize));
    network.add(new MaxSubsampleLayer(2));
    new Tester().init(samples, network, new EntropyLossLayer()).setVerbose(verbose).verifyConvergence(0.1,
        100);
  }

  @Test
  @org.junit.Ignore
  public void n2ActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 4 };
    final int[] outSize = inputSize;
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0, 0, 0 }), new Tensor(outSize, new double[] { 0.2, 0.3, 0.4, 0.1 }) } };
    final DAGNetwork net = new PipelineNetwork();
    net.add(new BiasLayer(inputSize));
    net.add(new L1NormalizationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void nestingLayer_feedback2() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.9, 0.1 }) } };
    final DAGNetwork net = new PipelineNetwork(); //
    net.add(new BiasLayer(inputSize)); //
    net.add(new SoftmaxActivationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  public void sigmoidActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.9, -.9 }) } };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(inputSize));
    network.add(new SigmoidActivationLayer());
    new Tester().init(samples, network, new EntropyLossLayer()).verifyConvergence(0.1, 100);
  }

  @Test
  @org.junit.Ignore
  public void softmaxActivationLayer_feedback() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Tensor[][] samples = new Tensor[][] { { new Tensor(inputSize, new double[] { 0, 0 }), new Tensor(outSize, new double[] { 0.1, 0.9 }) } };
    final DAGNetwork net = new PipelineNetwork();
    net.add(new BiasLayer(inputSize).setWeights(i -> Util.R.get().nextGaussian()));
    net.add(new SoftmaxActivationLayer());
    new Tester().init(samples, net, new EntropyLossLayer()).setVerbose(true) //
        // .setParallel(false)
        .verifyConvergence(0.1, 100);

  }


}
