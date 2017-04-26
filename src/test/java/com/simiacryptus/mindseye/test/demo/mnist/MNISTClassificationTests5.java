package com.simiacryptus.mindseye.test.demo.mnist;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.test.regression.MNISTClassificationTest;

public class MNISTClassificationTests5 extends MNISTClassificationTest {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    DAGNetwork net = new DAGNetwork();
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 10).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 10).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 10).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net = net.add(new SqActivationLayer());
    net = net.add(new SumSubsampleLayer(new int[] { 25, 25, 1 }));
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }

}
