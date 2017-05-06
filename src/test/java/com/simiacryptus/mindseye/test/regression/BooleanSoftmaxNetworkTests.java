package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;

public class BooleanSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSoftmaxNetworkTests.class);

  public static final Random random = new Random();

  public Tensor[][] getSoftmaxGateTrainingData(final BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Function<double[], double[]> fn = v -> new double[] { gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0, !gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0 };
    final Tensor[][] samples = new Tensor[][] {
        // XOR:
        { new Tensor(inputSize, new double[] { 0, 0 }), null }, { new Tensor(inputSize, new double[] { 0, 1 }), null }, { new Tensor(inputSize, new double[] { 1, 0 }), null },
        { new Tensor(inputSize, new double[] { 1, 1 }), null } };
    for (int i = 0; i < samples.length; i++) {
      samples[i][1] = new Tensor(outSize, fn.apply(samples[i][0].getData()));
    }
    return samples;
  }

  public void test(final Tensor[][] samples) {
    final int[] midSize = new int[] { 4 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    PipelineNetwork network = new PipelineNetwork();
    network.add(new DenseSynapseLayer(Tensor.dim(inputSize), midSize));
    network.add(new BiasLayer(midSize));
    network.add(new SigmoidActivationLayer());
    network.add(new DenseSynapseLayer(Tensor.dim(midSize), outSize));
    network.add(new BiasLayer(outSize));
    network.add(new SoftmaxActivationLayer());
    new Tester().init(samples,network,new EntropyLossLayer())
        .verifyConvergence(0.01, 100);
  }

  @Test
  public void test_AND() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final Tensor[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_OR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final Tensor[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_XOR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final Tensor[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

}
