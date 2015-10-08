package com.simiacryptus.mindseye.test.demo.mnist;

import com.simiacryptus.mindseye.LabeledObject;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;

public class MNISTClassificationTests5 extends MNISTClassificationTests {

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

  public boolean filter(final LabeledObject<NDArray> item) {
//    if (item.label.equals("[1]"))
//      return true;
//    if (item.label.equals("[8]"))
//      return true;
    return true;
  }

}
