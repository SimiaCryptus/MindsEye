package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;

public class MNISTClassificationTests4 extends MNISTClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28, 1 };
    DAGNetwork net = new DAGNetwork();

    net = net.add(new ConvolutionSynapseLayer(new int[] { 3, 3 }, 4).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net = net.add(new MaxSubsampleLayer(new int[] { 2, 2, 1 }));
    
    net = net.add(new SigmoidActivationLayer());
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 16).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net = net.add(new MaxSubsampleLayer(new int[] { 3, 3, 1 }));

    net = net.add(new SigmoidActivationLayer());
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 16).addWeights(() -> Util.R.get().nextGaussian() * .1));
    
    //net = net.add(new SumSubsampleLayer(new int[] { 4, 4, 1 }));
    
    int[] size = net.eval(new NDArray(inputSize)).data.getDims();
    net = net.add(new DenseSynapseLayer(NDArray.dim(size), new int[] { 10 }));
    net = net.add(new BiasLayer(10));
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }
  
}
