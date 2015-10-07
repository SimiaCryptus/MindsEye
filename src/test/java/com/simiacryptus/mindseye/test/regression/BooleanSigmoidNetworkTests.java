package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.test.Tester;

public class BooleanSigmoidNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSigmoidNetworkTests.class);

  public static final Random random = new Random();

  public NDArray[][] getTrainingData(final BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final Function<double[], double[]> fn = v -> new double[] { gate.apply(v[0] == 1, v[1] == 1) ? 1 : -1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 0 }), null }, { new NDArray(inputSize, new double[] { 0, 1 }), null }, { new NDArray(inputSize, new double[] { 1, 0 }), null },
        { new NDArray(inputSize, new double[] { 1, 1 }), null } };
    for (int i = 0; i < samples.length; i++) {
      samples[i][1] = new NDArray(outSize, fn.apply(samples[i][0].getData()));
    }
    return samples;
  }

  public void test(final NDArray[][] samples, int min) {
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    new Tester().init(samples, new DAGNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer())
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)).add(new BiasLayer(outSize)).add(new SigmoidActivationLayer()), (NNLayer<?>) new SqLossLayer())
        .verifyConvergence(0.01, 100, min);
  }

  @Test
  public void test_BasicNN_AND() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples, 95);
  }

  @Test
  public void test_BasicNN_OR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples, 95);
  }

  @Test
  public void test_BasicNN_XOR() throws Exception {
    final BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples, 20);
  }

}
