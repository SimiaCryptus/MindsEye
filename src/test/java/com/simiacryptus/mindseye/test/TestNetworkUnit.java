package com.simiacryptus.mindseye.test;

import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.PipelineNetwork;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;

public class TestNetworkUnit {
  static final Logger log = LoggerFactory.getLogger(TestNetworkUnit.class);

  public static final Random random = new Random();

  @Test
  public void test_BasicNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(inputSize))
    .add(new SigmoidActivationLayer())
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(outSize))
    .setRate(0.0001)
    .test(samples, 100000, 0.1, 10);
  }

  @Test
  public void test_BasicNN_AND() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 3 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(midSize))
    .add(new SigmoidActivationLayer())

    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(outSize))
    .add(new SigmoidActivationLayer())
    .setRate(0.0001)
    .test(samples, 100000, 0.01, 10);
  }

  @Test
  public void test_BasicNN_AND_Softmax() throws Exception {
    final int[] midSize = new int[] { 2 };
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 1, -1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1, -1 }) },
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1, 1 }) }
    };
    new PipelineNetwork()
//    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
//    .add(new BiasLayer(midSize))
//    .add(new SigmoidActivationLayer())

    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(outSize).setMass(2))
    .add(new SigmoidActivationLayer())
    //.add(new SoftmaxActivationLayer())
    .setRate(0.0001).setVerbose(true)
    .test(samples, 100000, 0.01, 10);
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
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 1 }) }
    };
    new PipelineNetwork()
    // Becomes unstable if these are added:
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(midSize))
    .add(new SigmoidActivationLayer())

    // Works okay:
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(outSize))
    .add(new SigmoidActivationLayer())
    .setRate(0.001)
    .test(samples, 100000, 0.01, 10);
  }

  @Test
  public void test_BasicNN_XOR() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] midSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        // XOR:
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 1 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { -1 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork()
    .add(
            new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).setMomentumDecay(0.5)
                .setMass(2.))
    .add(new BiasLayer(midSize).setMomentumDecay(0.5).setMass(2.))
    .add(new SigmoidActivationLayer())

    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).setMomentumDecay(0.9))
    .add(new BiasLayer(outSize))
    .add(new SigmoidActivationLayer())
    .setRate(0.01).setVerbose(true)
    .test(samples, 100000, 0.01, 10);
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
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { -1 }) }
    };
    new PipelineNetwork()
    .add(
            new DenseSynapseLayer(NDArray.dim(inputSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).setMass(5.)
                .setMomentumDecay(0.5))
    .add(new BiasLayer(midSize).setMass(5.).setMomentumDecay(0.5))
    .add(new SigmoidActivationLayer())
    .add(
            new DenseSynapseLayer(NDArray.dim(midSize), midSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).setMomentumDecay(0.8)
                .setMass(2.))
    .add(new BiasLayer(midSize).setMomentumDecay(0.8).setMass(2.))
    .add(new SigmoidActivationLayer())
    .add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).setMomentumDecay(0.9))
    .add(new BiasLayer(outSize).setMomentumDecay(0.9))
    .add(new SigmoidActivationLayer())
    .setRate(0.0001).setVerbose(true)
    .test(samples, 100000, 0.01, 10);
  }

  @Test
  public void test_LinearNN() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 1 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0 }) },
        { new NDArray(inputSize, new double[] { 0, 0 }), new NDArray(outSize, new double[] { 0.5 }) },
        { new NDArray(inputSize, new double[] { 1, 1 }), new NDArray(outSize, new double[] { 0 }) }
    };
    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new BiasLayer(outSize))
    .setRate(0.001)
    .test(samples, 100000, 0.1, 100);
  }

  @Test
  public void testDenseLinearLayer_2Layer() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };

    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .setRate(0.01)
    .test(samples, 10000, 0.01, 100);

    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).freeze())
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .setRate(0.01)
    .test(samples, 10000, 0.01, 100);

    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), inputSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()).freeze())
    .setRate(0.01)
    .test(samples, 10000, 0.01, 100);
  }

  @Test
  public void testDenseLinearLayer_Basic() throws Exception {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final NDArray[][] samples = new NDArray[][] {
        { new NDArray(inputSize, new double[] { 0, 1 }), new NDArray(outSize, new double[] { 1, 0 }) },
        { new NDArray(inputSize, new double[] { 1, 0 }), new NDArray(outSize, new double[] { 0, 1 }) }
    };

    new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize).addWeights(() -> 0.1 * TestNetworkUnit.random.nextGaussian()))
    .setRate(0.1)
    .test(samples, 1000, 0.01, 100);
  }

}
