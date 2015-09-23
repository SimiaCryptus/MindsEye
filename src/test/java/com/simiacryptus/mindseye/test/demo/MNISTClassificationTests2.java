package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests2 extends MNISTClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28 };
    final int[] midSize = new int[] { 20 };
    final int[] outSize = new int[] { 10 };
    DAGNetwork net = new DAGNetwork();

    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 8).addWeights(() -> Util.R.get().nextGaussian() * 1.));
    net = net.add(new MaxSubsampleLayer(new int[] { 3, 3, 1 }));

    int headSize = net.eval(new EvaluationContext(), new NDArray(inputSize)).data.dim();
    net = net.add(new DenseSynapseLayer(headSize, midSize));
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

  @Override
  public Tester buildTrainer(final NDArray[][] samples, final NNLayer<DAGNetwork> net) {
    //SqLossLayer lossLayer = new SqLossLayer();
    EntropyLossLayer lossLayer = new EntropyLossLayer();
    Tester trainer = new Tester().init(samples, net, lossLayer).setVerbose(true);
    trainer.setVerbose(true);
    trainer.trainingContext().setTimeout(1, java.util.concurrent.TimeUnit.HOURS);
    return trainer;
  }

}
