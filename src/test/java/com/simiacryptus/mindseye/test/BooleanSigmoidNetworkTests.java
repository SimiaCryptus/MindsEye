package com.simiacryptus.mindseye.test;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;

public class BooleanSigmoidNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSigmoidNetworkTests.class);
  
  public static final Random random = new Random();
  
  @Test
  public void test_BasicNN_XOR() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a != b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples);
  }
  
  @Test
  public void test_BasicNN_AND() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a && b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples);
  }
  
  @Test
  public void test_BasicNN_OR() throws Exception {
    BiFunction<Boolean, Boolean, Boolean> gate = (a, b) -> a || b;
    final NDArray[][] samples = getTrainingData(gate);
    test(samples);
  }
  
  public void test(final NDArray[][] samples) {
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    boolean verbose = false;
    new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)
            .setHalflife(4)
            .setMass(5.))
         .add(new BiasLayer(midSize)
         .setHalflife(3)
         .setMass(5.)
         )
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)
        // .setVerbose(verbose)
        )
        .add(new BiasLayer(outSize).setMass(2.))
        .add(new SigmoidActivationLayer()
        // .setVerbose(verbose)
        )
        .setMutationAmplitude(3)
        .trainer(samples)
        .setMutationAmount(.1)
        .setVerbose(verbose)
        .setStaticRate(5.)
        .setDynamicRate(0.01)
        .setMaxDynamicRate(0.1)
        .setMinDynamicRate(0.)
        .setImprovementStaleThreshold(5)
        .setLoopA(5)
        .setLoopB(5)
        
        .verifyConvergence(10000, 0.01, 1);
  }
  
  public NDArray[][] getTrainingData(BiFunction<Boolean, Boolean, Boolean> gate) {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    Function<double[], double[]> fn = v -> new double[] { gate.apply(v[0] == 1, v[1] == 1) ? 1 : -1 };
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
