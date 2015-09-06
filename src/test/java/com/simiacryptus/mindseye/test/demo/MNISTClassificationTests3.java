package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.MaxSubsampleLayer;
import com.simiacryptus.mindseye.layers.MinMaxFilterLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests3 extends MNISTClassificationTests {
  
  @Override
  public PipelineNetwork buildNetwork() {
    final int[] inputSize = new int[] { 28, 28 };
    final int[] outSize = new int[] { 10 };
    PipelineNetwork net = new PipelineNetwork();
    
    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 3).addWeights(()->Util.R.get().nextGaussian()*1.));
    net = net.add(new MaxSubsampleLayer(new int[] { 27, 27, 1 }));
    
    net = net.add(new DenseSynapseLayer(net.eval(new NDArray(inputSize)).data.dim(), outSize));
    net = net.add(new BiasLayer(outSize));
    
    // net = net.add(new ExpActivationLayer())
    // net = net.add(new L1NormalizationLayer());
    // net = net.add(new LinearActivationLayer());
    net = net.add(new MinMaxFilterLayer());
    net = net.add(new SoftmaxActivationLayer());
    return net;
  }
  
}