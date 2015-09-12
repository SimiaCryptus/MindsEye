package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests2 extends MNISTClassificationTests {

  @Override
  public DAGNetwork buildNetwork() {
    final int[] inputSize = new int[] { 28, 28 };
    final int[] midSize = new int[] { 20 };
    final int[] outSize = new int[] { 10 };
    DAGNetwork net = new DAGNetwork();

    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 8).addWeights(() -> Util.R.get().nextGaussian() * 1.));
    net = net.add(new MaxSubsampleLayer(new int[] { 3, 3, 1 }));

    net = net.add(new DenseSynapseLayer(net.eval(new NDArray(inputSize)).data.dim(), midSize));
    net = net.add(new BiasLayer(midSize));

    net = net.add(new MinMaxFilterLayer());
    net = net.add(new SigmoidActivationLayer());

    net = net.add(new DenseSynapseLayer(NDArray.dim(midSize), outSize));
    net = net.add(new BiasLayer(outSize));

    // net = net.add(new ExpActivationLayer())
    // net = net.add(new L1NormalizationLayer());
    // net = net.add(new LinearActivationLayer());
    net = net.add(new MinMaxFilterLayer());
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }

}
