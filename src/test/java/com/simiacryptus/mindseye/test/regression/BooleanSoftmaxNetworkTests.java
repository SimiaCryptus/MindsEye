package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.test.Tester;

public class BooleanSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSoftmaxNetworkTests.class);

  public static final Random random = new Random();

  public NDArray[][] getSoftmaxGateTrainingData(final BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final Function<double[], double[]> fn = v -> new double[] { gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0, !gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 0 }), null }, { new NDArray(inputSize, new double[] { 0, 1 }), null }, { new NDArray(inputSize, new double[] { 1, 0 }), null },
        { new NDArray(inputSize, new double[] { 1, 1 }), null } };
    for (int i = 0; i < samples.length; i++) {
      samples[i][1] = new NDArray(outSize, fn.apply(samples[i][0].getData()));
    }
    return samples;
  }

  public void test(final NDArray[][] samples) {
    final int[] midSize = new int[] { 4 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    new Tester()
        .init(samples,
            new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer())
                .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)).add(new BiasLayer(outSize)).add(new SoftmaxActivationLayer().setVerbose(false)),
            new EntropyLossLayer())
        .verifyConvergence(0.01, 100);
  }

  @Test
  public void test_AND() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_OR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_XOR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

}
