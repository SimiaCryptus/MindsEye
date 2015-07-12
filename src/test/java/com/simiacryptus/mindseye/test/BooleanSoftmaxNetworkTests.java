package com.simiacryptus.mindseye.test;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;

public class BooleanSoftmaxNetworkTests {
  static final Logger log = LoggerFactory.getLogger(BooleanSoftmaxNetworkTests.class);
  
  public static final Random random = new Random();
  
  
  // @Ignore
  @Test
  public void test_BasicNN_XOR_Softmax() throws Exception {
    final NDArray[][] samples = getSoftmaxGateTrainingData((a, b) -> a && b);
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    new PipelineNetwork()
        
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))
        .add(new BiasLayer(inputSize))
        .add(new SigmoidActivationLayer())
        
        .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SoftmaxActivationLayer())
        
        .trainer(samples)
        .setVerbose(true)
        .setMaxDynamicRate(100)
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
