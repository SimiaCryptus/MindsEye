package com.simiacryptus.mindseye.test.regression;

import java.util.Random;
import java.util.function.DoubleSupplier;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.test.Tester;

public class SimpleNetworkTests {
  static final Logger log = LoggerFactory.getLogger(SimpleNetworkTests.class);

  public static final Random random = new Random();

  @Test
  public void test_BasicNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) } };
    new Tester().init(samples, new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)).add(new BiasLayer(inputSize)).add(new SigmoidActivationLayer())
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)).add(new BiasLayer(outSize)), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.1, 10);
  }

  @Test
  public void test_BasicNN_OR() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) } };
    new Tester().init(samples, new DAGNetwork()//
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize))//
    .add(new BiasLayer(midSize))//
    .add(new SigmoidActivationLayer())//
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize))//
    .add(new BiasLayer(outSize))//
    .add(new SigmoidActivationLayer()), (NNLayer<?>) new SqLossLayer()).verifyConvergence(0.01, 100);
  }

  @Test
  public void test_BasicNN_XOR_3layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 4 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) } };
    new Tester().init(samples, new DAGNetwork()
    
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer())
    
    .add(new DenseSynapseLayer(NDArray.dim(midSize), midSize)).add(new BiasLayer(midSize)).add(new SigmoidActivationLayer())
    
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)).add(new BiasLayer(outSize)).add(new SigmoidActivationLayer()), (NNLayer<?>) new EntropyLossLayer())
        .verifyConvergence(0.01, 100);
  }

  @Test
  public void test_DualSigmoid() throws Exception {
    final int[] inputSize = new int[] { 1 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { -1 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0 }), new NDArray(outSize, new double[] { .2 }) },
        { new NDArray(inputSize, new double[] { 1 }), new NDArray(outSize, new double[] { 0 }) } };
    new Tester().init(samples, new DAGNetwork()//
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).setWeights(new double[] { 1, 1 }).freeze())//
    .add(new BiasLayer(midSize).set(new double[] { -1, 1 }))//
    .add(new SigmoidActivationLayer())//
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).setWeights(new double[] { 1, -1 }).freeze()), (NNLayer<?>) new EntropyLossLayer())//
        .verifyConvergence(0.1, 10);
  }

  @Test
  public void test_LinearNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 3 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] { { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) } };
    new Tester().init(samples, new DAGNetwork().add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize)).add(new DenseSynapseLayer(NDArray.dim(midSize), midSize))
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize)).add(new BiasLayer(outSize)), (NNLayer<?>) new EntropyLossLayer()).setVerbose(false).verifyConvergence(0.1, 100);
  }

  @Test
  public void testDenseLinearLayer_2Layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] { 
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) } 
      };

    final DoubleSupplier f = () -> {
      double x = 15.0 * SimpleNetworkTests.random.nextGaussian();
      if (x < 0 && x > -0.001) {
        x -= 1;
      }
      if (x >= 0 && x < 0.001) {
        x += 1;
      }
      return x;
    };
    new Tester().init(samples, new DAGNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(f).freeze()) //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)), (NNLayer<?>) new EntropyLossLayer())//
        //.setVerbose(true)//
        .setStaticRate(0.01)//
        .verifyConvergence(0.01, 1);

    new Tester().init(samples, new DAGNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(f).freeze()), (NNLayer<?>) new EntropyLossLayer())//
        //.setStaticRate(0.01)
        // .setVerbose(true).setParallel(false)
        .verifyConvergence(0.01, 100, 90);

    new Tester().init(samples, new DAGNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)), (NNLayer<?>) new EntropyLossLayer()).setStaticRate(0.01).verifyConvergence(0.01, 100, 80);

    new Tester().init(samples, new DAGNetwork() //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize)) //
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize)), (NNLayer<?>) new EntropyLossLayer()).verifyConvergence(0.01, 100, 50);
  }

}
