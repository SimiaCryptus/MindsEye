package com.simiacryptus.mindseye.test.demo;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.training.NetInitializer;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests2 extends MNISTClassificationTests {

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    //final int[] inputSize = new int[] { 28, 28, 1 };
    //final int[] midSize = new int[] { 20 };
    //final int[] outSize = new int[] { 10 };
    DAGNetwork net = new DAGNetwork();

    net = net.add(new ConvolutionSynapseLayer(new int[] { 2, 2 }, 10).addWeights(() -> Util.R.get().nextGaussian() * .1));
    //int headSize = new NDArray(inputSize).getData().length;
    net = net.add(new SumSubsampleLayer(new int[] { 27, 27, 1 }));

    //int headSize = net.eval(new EvaluationContext(), new NDArray(inputSize)).data.dim();
    //net = net.add(new DenseSynapseLayer(headSize, outSize).addWeights(() -> Util.R.get().nextGaussian() * .005));

//    net = net.add(new BiasLayer(midSize).addWeights(() -> Util.R.get().nextGaussian() * .1));
//    net = net.add(new MinMaxFilterLayer());
//    net = net.add(new SigmoidActivationLayer());

    //net = net.add(new DenseSynapseLayer(NDArray.dim(midSize), outSize).addWeights(() -> Util.R.get().nextGaussian() * .05));
    //net = net.add(new BiasLayer(outSize).addWeights(() -> Util.R.get().nextGaussian() * .1));

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
    Tester trainer = new Tester(){
      
      @Override
      public NetInitializer getInitializer() {
        NetInitializer netInitializer = new NetInitializer();
        netInitializer.setAmplitude(0.);
        return netInitializer;
      }

    }.init(samples, net, lossLayer).setVerbose(true);
    trainer.setVerbose(true);
    trainer.trainingContext().setTimeout(30, java.util.concurrent.TimeUnit.MINUTES);
    return trainer;
  }

}
