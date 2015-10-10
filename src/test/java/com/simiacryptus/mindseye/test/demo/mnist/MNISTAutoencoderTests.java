package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;
import com.simiacryptus.mindseye.test.dev.MNIST;
import com.simiacryptus.mindseye.test.dev.SimpleMNIST;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;

public class MNISTAutoencoderTests {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public NNLayer<DAGNetwork> buildNetwork() {
    final int[] inputSize = new int[] { 28, 28, 1 };
    final int[] midSize = new int[] { 1000 };
    DAGNetwork net = new DAGNetwork();
    net = net.add(new DenseSynapseLayer(NDArray.dim(inputSize), midSize).setWeights(()->Util.R.get().nextGaussian()*0.1));
    net = net.add(new BiasLayer(midSize));
    //net = net.add(new SigmoidActivationLayer());
    net = net.add(new DenseSynapseLayer(NDArray.dim(midSize), inputSize).setWeights(()->Util.R.get().nextGaussian()*0.1));
    net = net.add(new BiasLayer(inputSize));
    //net = net.add(new BiasLayer(inputSize));
    return net;
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

  @Test
  public void test() throws Exception {
    final int hash = Util.R.get().nextInt();
    log.debug(String.format("Shuffle hash: 0x%s", Integer.toHexString(hash)));
    final NDArray[][] trainingData = transformDataSet(MNIST.trainingDataStream(), 100, hash);
    final NDArray[][] validationData = transformDataSet(MNIST.validationDataStream(), 100, hash);
    final NNLayer<DAGNetwork> net = buildNetwork();
    final Map<BufferedImage, String> report = new HashMap<>();
    final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler = (n, trainingContext) -> {
      evaluateImageList(n, trainingData, net.id).stream().forEach(i->report.put(i, ""));
      evaluateImageList(n, validationData, net.id).stream().forEach(i->report.put(i, ""));
      return null;
    };
    try {
      Tester tester = new Tester();
      tester.setVerbose(true);
      GradientDescentTrainer trainer = tester.getGradientDescentTrainer();
      DAGNetwork supervisedNetwork = Tester.supervisionNetwork(net, new SqLossLayer());
      trainer.setNet(supervisedNetwork);
      trainer.setData(trainingData);
      tester.handler.add(resultHandler);
      tester.trainingContext().setTimeout(10, java.util.concurrent.TimeUnit.MINUTES);
      tester.verifyConvergence(0.1, 1);
    } finally {
      final Stream<String> map = report.entrySet().stream().map(e -> Util.toInlineImage(e.getKey(), e.getValue().toString()));
      Util.report(map.toArray(i -> new String[i]));
    }
  }

  public List<BufferedImage> evaluateImageList(DAGNetwork n, final NDArray[][] validationData, UUID id) {
    final NNLayer<?> mainNetwork = n.getChild(id);
    final List<BufferedImage> img = java.util.Arrays.stream(validationData).map(array->{
      final NNResult output = mainNetwork.eval(array);
      return Util.toImage(output.data);
    }).collect(java.util.stream.Collectors.toList());
    return img;
  }

  public NDArray[][] transformDataSet(Stream<LabeledObject<NDArray>> trainingDataStream, int limit, final int hash) {
    return trainingDataStream
        .collect(java.util.stream.Collectors.toList()).stream().parallel()
        .filter(this::filter)
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .limit(limit)
        .map(obj -> new NDArray[] { obj.data, obj.data })
        .toArray(i1 -> new NDArray[i1][]);
  }

}
