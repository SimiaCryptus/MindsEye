package com.simiacryptus.mindseye.test.demo.mnist;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.test.regression.MNISTClassificationTest;

public class MNISTClassificationTests2 extends MNISTClassificationTest {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    DAGNetwork net = new PipelineNetwork();
    final int n = 2;
    final int m = 28 - n + 1;
    net.add(new ConvolutionSynapseLayer(new int[] { n, n }, 10).addWeights(() -> Util.R.get().nextGaussian() * .001));
    net.add(new SqActivationLayer());
    net.add(new SumSubsampleLayer(new int[] { m, m, 1 }));
    net.add(new SoftmaxActivationLayer());
    return net;
  }

}
