package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.training.PipelineNetwork;

public class BooleanSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSoftmaxNetworkTests.class);
  
  public static final Random random = new Random();

  @Test
  public void test_AND() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  @Test
  public void test_OR() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }
  
  @Test
  public void test_XOR() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final NDArray[][] samples = getSoftmaxGateTrainingData(gate);
    test(samples);
  }

  public void test(final NDArray[][] samples) {
    final int[] midSize = new int[] { 4 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)
            .setHalflife(4)
            .setMass(5.))
        .add(new BiasLayer(midSize)
            .setHalflife(4)
            .setMass(5.))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SoftmaxActivationLayer().setVerbose(false))

        
        .setMutationAmplitude(3)
        .trainer(samples)
        .setMutationAmount(1)
        //.setVerbose(true)
        .setStaticRate(.5)
        .setDynamicRate(0.5)
        .setMaxDynamicRate(5000)
        .setMinDynamicRate(1e-9)
        .setImprovementStaleThreshold(10)

        .verifyConvergence(10000, 0.01, 1);
  }
  
  public NDArray[][] getSoftmaxGateTrainingData(BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    Function<double[], double[]> fn = v -> new double[] { 
        gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0, 
        !gate.apply(v[0] == 1, v[1] == 1) ? 1 : 0 
    };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 0 }), null },
        { new NDArray(inputSize, new double[] { 0, 1 }), null },
        { new NDArray(inputSize, new double[] { 1, 0 }), null },
        { new NDArray(inputSize, new double[] { 1, 1 }), null }
    };
    for (int i = 0; i < samples.length; i++)
      samples[i][1] = new NDArray(outSize, fn.apply(samples[i][0].data));
    return samples;
  }
  
}
