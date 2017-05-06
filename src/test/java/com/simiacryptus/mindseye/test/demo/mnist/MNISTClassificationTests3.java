package com.simiacryptus.mindseye.test.demo.mnist;

import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.regression.MNISTClassificationTest;

public class MNISTClassificationTests3 extends MNISTClassificationTest {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28, 1 };
    DAGNetwork net = new PipelineNetwork();

    net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 1).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 1).addWeights(() -> Util.R.get().nextGaussian() * .1));
    net.add(new MaxSubsampleLayer(new int[] { 2, 2, 1 }));
    net.add(new SigmoidActivationLayer());

    // net.add(new SumSubsampleLayer(new int[] { 13, 13, 1 }));

    final int[] size = net.eval(new Tensor(inputSize)).data[0].getDims();
    net.add(new DenseSynapseLayer(Tensor.dim(size), new int[] { 10 }));
    net.add(new BiasLayer(10));
    net.add(new SoftmaxActivationLayer());
    return net;
  }

  @Override
  public Tester buildTrainer(Tensor[][] samples, NNLayer<DAGNetwork> net) {
    // TODO Auto-generated method stub
    Tester trainer = super.buildTrainer(samples, net);
    trainer.getGradientDescentTrainer().setTrainingSize(1000);
    return trainer;
  }

}
