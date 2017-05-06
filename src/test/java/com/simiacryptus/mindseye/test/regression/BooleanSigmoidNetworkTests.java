package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.simiacryptus.mindseye.net.SupervisedNetwork;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;

public class BooleanSigmoidNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSigmoidNetworkTests.class);

  public static final Random random = new Random();

  public Tensor[][] getTrainingData(final BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final Function<double[], double[]> fn = v -> new double[] { gate.apply(v[0] == 1, v[1] == 1) ? 1 : -1 };
    final Tensor[][] samples = new Tensor[][] {
        // XOR:
        { new Tensor(inputSize, new double[] { 0, 0 }), null },
        { new Tensor(inputSize, new double[] { 0, 1 }), null },
        { new Tensor(inputSize, new double[] { 1, 0 }), null },
        { new Tensor(inputSize, new double[] { 1, 1 }), null } };
    for (int i = 0; i < samples.length; i++) {
      samples[i][1] = new Tensor(outSize, fn.apply(samples[i][0].getData()));
    }
    return samples;
  }

  public void test(final Tensor[][] samples) {
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    PipelineNetwork net = new PipelineNetwork();
    net.add(new DenseSynapseLayer(Tensor.dim(inputSize), midSize));
    net.add(new BiasLayer(midSize));
    net.add(new SigmoidActivationLayer());
    net.add(new DenseSynapseLayer(Tensor.dim(midSize), outSize));
    net.add(new BiasLayer(outSize));
      net.add(new SigmoidActivationLayer());
    SupervisedNetwork supervisedNetwork = new SupervisedNetwork(net, new SqLossLayer());
    Tester init = new Tester();
    GradientDescentTrainer trainer = init.getGradientDescentTrainer();
    trainer.setNet(supervisedNetwork);
    trainer.setData(samples);
    init.verifyConvergence(0.01, 100);
  }

  @Test
  public void test_BasicNN_AND() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final Tensor[][] samples = getTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_BasicNN_OR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final Tensor[][] samples = getTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_BasicNN_XOR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final Tensor[][] samples = getTrainingData(gate);
    test(samples);
  }

}
