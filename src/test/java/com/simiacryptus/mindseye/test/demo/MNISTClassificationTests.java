package com.simiacryptus.mindseye.test.demo;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Test;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.dev.MNIST;
import com.simiacryptus.mindseye.test.dev.SimpleMNIST;
import com.simiacryptus.mindseye.training.DevelopmentTrainer;
import com.simiacryptus.mindseye.training.DynamicRateTrainer;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;
import com.simiacryptus.mindseye.training.NetInitializer;
import com.simiacryptus.mindseye.util.LabeledObject;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests extends ClassificationTestBase {

  public MNISTClassificationTests() {
    super();
    this.drawBG = false;
  }

  @Override
  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28 };
    final int[] outSize = new int[] { 10 };
    DAGNetwork net = new DAGNetwork();
    net = net.add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize));
    net = net.add(new BiasLayer(outSize));
    //net = net.add(new MinMaxFilterLayer());
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
        netInitializer.setAmplitude(0.1);
        return netInitializer;
      }

    }.init(samples, net, lossLayer).setVerbose(true);
    trainer.setVerbose(true);
    trainer.trainingContext().setTimeout(2, java.util.concurrent.TimeUnit.MINUTES);
    return trainer;
  }

  public boolean filter(final LabeledObject<NDArray> item) {
    if (item.label.equals("[0]"))
      return true;
    if (item.label.equals("[5]"))
      return true;
    if (item.label.equals("[9]"))
      return true;
    return true;
  }

  @Override
  public double[] inputToXY(final NDArray input, final int classificationActual, final int classificationExpected) {
    final double n = 10.;
    final double[] c = new double[] { //
        (classificationActual + Util.R.get().nextDouble()) / (n + 1), //
        (classificationExpected + Util.R.get().nextDouble()) / (n + 1) //
    };
    return new double[] { c[0] * 6 - 3, c[1] * 6 - 3 };
  }

  private String remap(final String label) {
    switch (label) {
    // case "[0]":
    // return "[5]";
    // case "[5]":
    // return "[9]";
    // case "[9]":
    // return "[0]";
    default:
      return label;
    }
  }

  @Test
  public void test() throws Exception {
    test(trainingData());
  }

  public NDArray[][] trainingData() throws IOException {
    final int maxSize = 1000;
    final List<LabeledObject<NDArray>> data = Util.shuffle(MNIST.trainingDataStream().filter(this::filter).collect(Collectors.toList()));
    final NDArray[][] trainingData = data.parallelStream().limit(maxSize).map(obj -> {
      final int out = SimpleMNIST.toOut(remap(obj.label));
      final NDArray output = SimpleMNIST.toOutNDArray(out, 10);
      return new NDArray[] { obj.data, output };
    }).toArray(i -> new NDArray[i][]);
    return trainingData;
  }

  @Override
  public void verify(final Tester trainer) {
    trainer.verifyConvergence(0.0, 1);
  }

}
