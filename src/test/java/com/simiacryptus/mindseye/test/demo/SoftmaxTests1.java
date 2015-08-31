package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.PipelineNetwork;

public class SoftmaxTests1 extends SimpleClassificationTests {
  @Override
  public PipelineNetwork buildNetwork() {
    final int[] inputSize = new int[] { 2 };
    final int[] outSize = new int[] { 2 };
    final PipelineNetwork net = new PipelineNetwork()
    .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
    .add(new BiasLayer(outSize))
    .add(new SoftmaxActivationLayer());
    return net;
  }

}
