package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.stream.Stream;

import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.test.MNIST;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.mindseye.training.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerGPU;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;

public class MNISTAutoencoderTestsGPU {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public NNLayer<DAGNetwork> buildNetwork() {
    List<int[]> sizes = new ArrayList<>();
    sizes.add(new int[] { 28, 28, 1 });
    sizes.add(new int[] { 1000 });
    //sizes.add(new int[] { 100 });
    List<Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU>> codecs = new ArrayList<>();
    for(int i=1;i<sizes.size();i++) {
      codecs.add(createCodecPair(sizes.get(i-1), sizes.get(i)));
    }
    return stackedCodecNetwork(codecs);
  }

  public Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU> createCodecPair(final int[] outerSize, final int[] innerSize) {
    DenseSynapseLayerGPU encode = new DenseSynapseLayerGPU(Tensor.dim(outerSize), innerSize).setWeights(()->Util.R.get().nextGaussian()*0.1);
    DenseSynapseLayerGPU decode = new DenseSynapseLayerGPU(Tensor.dim(innerSize), outerSize).setWeights((Coordinate c)->{
      int[] traw = new int[]{c.coords[1],c.coords[0]};
      int tindex = encode.getWeights().index(traw);
      Coordinate transposed = new Coordinate(tindex, traw);
      return encode.getWeights().get(transposed);
    });
    Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU> codec = new Tuple2<DenseSynapseLayerGPU,DenseSynapseLayerGPU>(encode, decode);
    return codec;
  }

  private DAGNetwork stackedCodecNetwork(List<Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU>> codecs) {
    DAGNetwork net = new DAGNetwork();
    for(int i=0;i<codecs.size();i++) {
      Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU> t = codecs.get(i);
      DenseSynapseLayerGPU encode = t.getFirst();
      net = net.add(encode);
      net = net.add(new BiasLayer(encode.outputDims));
      net = net.add(new SigmoidActivationLayer());
    }
    for(int i=codecs.size()-1;i>=0;i--) {
      Tuple2<DenseSynapseLayerGPU, DenseSynapseLayerGPU> t = codecs.get(i);
      DenseSynapseLayerGPU decode = t.getSecond();
      net = net.add(decode);
      net = net.add(new BiasLayer(decode.outputDims));
    }
    return net;
  }

  public boolean filter(final LabeledObject<Tensor> item) {
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
    final Tensor[][] trainingData = transformDataSet(MNIST.trainingDataStream(), 100000, hash);
    final Tensor[][] validationData = transformDataSet(MNIST.validationDataStream(), 100, hash);
    final NNLayer<DAGNetwork> net = buildNetwork();
    final List<String> report = new java.util.ArrayList<>();
    final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler = (trainedNetwork, trainingContext) -> {
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, java.util.Arrays.copyOf(trainingData, 100), net.id).stream().forEach(i->report.add(Util.toInlineImage(i, "TRAINING")));
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, validationData, net.id).stream().forEach(i->report.add(Util.toInlineImage(i, "TEST")));
      report.add("<hr/>");
      return null;
    };
    try {
      {
        getTester(net, java.util.Arrays.copyOf(trainingData, 10), resultHandler).trainTo(100);
        getTester(net, java.util.Arrays.copyOf(trainingData, 50), resultHandler).trainTo(10);
        getTester(net, java.util.Arrays.copyOf(trainingData, 100), resultHandler).trainTo(1);
//        getTester(net, java.util.Arrays.copyOf(trainingData, 40), resultHandler).verifyConvergence(1, 1);
//        getTester(net, java.util.Arrays.copyOf(trainingData, 50), resultHandler).verifyConvergence(.1, 1);
        //getTester(net, java.util.Arrays.copyOf(trainingData, 100), resultHandler).verifyConvergence(.1, 1);
//        getTester(net, trainingData, resultHandler).verifyConvergence(1, 1);
      }
    } finally {
      Util.report(report.stream().toArray(i -> new String[i]));
    }
  }

  public Tester getTester(final NNLayer<DAGNetwork> net, Tensor[][] trainingData2, final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler) {
    Tester tester = new Tester();
    tester.setVerbose(true);
    GradientDescentTrainer trainer = tester.getGradientDescentTrainer();
    DAGNetwork supervisedNetwork = Tester.supervisionNetwork(net, new SqLossLayer());
    trainer.setNet(supervisedNetwork);
    trainer.setData(trainingData2);
    if (null != resultHandler) {
      tester.handler.add(resultHandler);
    }
    tester.trainingContext().setTimeout(10, java.util.concurrent.TimeUnit.MINUTES);
    return tester;
  }

  public List<BufferedImage> evaluateImageList(DAGNetwork n, final Tensor[][] validationData, UUID id) {
    final NNLayer<?> mainNetwork = n.getChild(id);
    return java.util.Arrays.stream(validationData)
        .map(x->mainNetwork.eval(x).data[0])
        .map(x->Util.toImage(x))
        .collect(java.util.stream.Collectors.toList());
  }

  public Tensor[][] transformDataSet(Stream<LabeledObject<Tensor>> trainingDataStream, int limit, final int hash) {
    return trainingDataStream
        .collect(java.util.stream.Collectors.toList()).stream().parallel()
        .filter(this::filter)
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .limit(limit)
        .map(obj -> new Tensor[] { obj.data, obj.data })
        .toArray(i1 -> new Tensor[i1][]);
  }

}
