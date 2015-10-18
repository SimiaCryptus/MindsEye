package com.simiacryptus.mindseye.test.demo.mnist;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.test.regression.MNISTClassificationTest;

public class MNISTClassificationTests2 extends MNISTClassificationTest {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    DAGNetwork net = new DAGNetwork();
    final int n = 2;
    final int m = 28 - n + 1;
    net = net.add(new ConvolutionSynapseLayer(new int[] { n, n }, 10).addWeights(() -> Util.R.get().nextGaussian() * .001));
    net = net.add(new SqActivationLayer());
    net = net.add(new SumSubsampleLayer(new int[] { m, m, 1 }));
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }

}
