package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.IntFunction;
import java.util.stream.Stream;

import com.simiacryptus.util.test.MNIST;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.ml.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.DAGNode;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.net.meta.AvgMetaLayer;
import com.simiacryptus.mindseye.net.meta.Sparse01MetaLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.reducers.SumReducerLayer;
import com.simiacryptus.mindseye.net.util.VerboseWrapper;
import com.simiacryptus.mindseye.net.util.WeightExtractor;
import com.simiacryptus.mindseye.test.Tester;
import com.simiacryptus.mindseye.test.demo.ClassificationTestBase;
import com.simiacryptus.mindseye.training.GradientDescentTrainer;

public class MNISTAutoencoderTests2 {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public static class AutoencodingNetwork {
    DAGNetwork     net;
    DAGNode        center;
    DAGNode        feedback;

    public AutoencodingNetwork() {
      this(new int[] { 28, 28, 1 }, new int[] { 100 });
    }

    public AutoencodingNetwork(int[] outerSize, int[] innerSize) {
      super();
      this.net = new DAGNetwork();
      List<NNLayer<?>> weightNormalizationList = new ArrayList<>();
      DenseSynapseLayerJBLAS encode = new DenseSynapseLayerJBLAS(NDArray.dim(outerSize), innerSize);// .setWeights(()->Util.R.get().nextGaussian()*0.1);
      weightNormalizationList.add(encode);
      {
        net = net.add(encode);
        BiasLayer bias = new BiasLayer(encode.outputDims);
        weightNormalizationList.add(bias);
        net = net.add(bias);
        net = net.add(new SigmoidActivationLayer().setBalanced(false));
      }
      center = net.getHead();
      {
        DenseSynapseLayerJBLAS decode = new DenseSynapseLayerJBLAS(NDArray.dim(innerSize), outerSize)
            .setWeights((Coordinate c) -> {
              int[] traw = new int[] { c.coords[1], c.coords[0] };
              int tindex = encode.getWeights().index(traw);
              Coordinate transposed = new Coordinate(tindex, traw);
              double foo = encode.getWeights().get(transposed);
              return foo;
            });
        weightNormalizationList.add(decode);
        net = net.add(decode);
        BiasLayer bias = new BiasLayer(decode.outputDims);
        weightNormalizationList.add(bias);
        net = net.add(bias);
        // net = net.add(new SigmoidActivationLayer());
        feedback = net.getHead();
      }
      List<DAGNode> fitnessSet = new ArrayList<>();
      fitnessSet.add(this.net
          .add(new SqLossLayer(), this.feedback, this.net.getInput().get(0))
          .add(new AvgMetaLayer()).getHead());
      fitnessSet.add(this.net.add(new Sparse01MetaLayer(), this.center)
          .add(new SumReducerLayer())
          .add(new LinearActivationLayer().setWeight(.1).freeze()).getHead());
      fitnessSet.add(this.net.add(new SumInputsLayer(), weightNormalizationList.stream().map(x->net
          .add(new WeightExtractor(0, x), new DAGNode[] {})
          .add(new SqActivationLayer())
          .add(new SumReducerLayer()).getHead()).toArray(i->new DAGNode[i]))
          .add(new LinearActivationLayer().setWeight(0.001).freeze())
          .getHead());
      this.net.add(new VerboseWrapper("sums", new SumInputsLayer()), fitnessSet.toArray(new DAGNode[] {}));
    }

  }

  @Test
  public void test() throws Exception {
    final int hash = Util.R.get().nextInt();
    log.debug(String.format("Shuffle hash: 0x%s", Integer.toHexString(hash)));
    final NDArray[][] trainingData = transformDataSet(MNIST.trainingDataStream(), 100000, hash);
    final NDArray[][] validationData = transformDataSet(MNIST.validationDataStream(), 100, hash);
    final AutoencodingNetwork net = new AutoencodingNetwork();
    final List<String> report = new java.util.ArrayList<>();
    final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler = (trainedNetwork, trainingContext) -> {
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, java.util.Arrays.copyOf(trainingData, 100), net.feedback).stream()
          .forEach(i -> report.add(Util.toInlineImage(i, "TRAINING")));
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, validationData, net.feedback).stream()
          .forEach(i -> report.add(Util.toInlineImage(i, "TEST")));
      report.add("<hr/>");
      return null;
    };
    try {
      {
        for (int i = 0; i < 10; i++)
          getTester(net, select(trainingData, 100), resultHandler).trainTo(.1);
        // getTester(net, select(trainingData, 1000),
        // resultHandler).trainTo(.0);
        // getTester(net, java.util.Arrays.copyOf(trainingData, 40),
        // resultHandler).verifyConvergence(1, 1);
        // getTester(net, java.util.Arrays.copyOf(trainingData, 50),
        // resultHandler).verifyConvergence(.1, 1);
        // getTester(net, java.util.Arrays.copyOf(trainingData, 100),
        // resultHandler).verifyConvergence(.1, 1);
        // getTester(net, trainingData, resultHandler).verifyConvergence(1, 1);
      }
    } finally {
      Util.report(report.stream().toArray(i -> new String[i]));
    }
  }

  private static NDArray[][] select(NDArray[][] array, int n) {
    return select(array, n, i -> new NDArray[i][]);
  }

  private static <T> T[] select(T[] array, int n, IntFunction<T[]> generator) {
    return selectStream(array, n).toArray(generator);
  }

  public static <T> Stream<T> selectStream(T[] array, int n) {
    return shuffledStream(array).limit(n);
  }

  public static <T> Stream<T> shuffledStream(T[] array) {
    return shuffledStream(array, Util.R.get().nextInt());
  }

  public static <T> Stream<T> shuffledStream(T[] array, int hash) {
    return java.util.Arrays.stream(array)
        .sorted(java.util.Comparator.comparingInt(x -> System.identityHashCode(x) ^ hash));
  }

  public Tester getTester(final AutoencodingNetwork codec, NDArray[][] trainingData,
      final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler) {
    Tester tester = new Tester();
    tester.setVerbose(true);
    GradientDescentTrainer trainer = tester.getGradientDescentTrainer();
    trainer.setNet(codec.net);
    trainer.setData(trainingData);
    if (null != resultHandler) {
      tester.handler.add(resultHandler);
    }
    tester.trainingContext().setTimeout(30, java.util.concurrent.TimeUnit.SECONDS);
    return tester;
  }

  public List<BufferedImage> evaluateImageList(DAGNetwork n, final NDArray[][] validationData, DAGNode feedback) {
    // final NNLayer<?> mainNetwork = n.getChild(feedback);
    return java.util.Arrays.stream(validationData).map(x -> feedback.get(n.buildExeCtx(x)).data[0])
        .map(x -> Util.toImage(x)).collect(java.util.stream.Collectors.toList());
  }

  public NDArray[][] transformDataSet(Stream<LabeledObject<NDArray>> trainingDataStream, int limit, final int hash) {
    return trainingDataStream.collect(java.util.stream.Collectors.toList()).stream().parallel()
        .sorted(java.util.Comparator.comparingInt(obj -> 0xEFFFFFFF & (System.identityHashCode(obj) ^ hash)))
        .limit(limit).map(obj -> new NDArray[] { obj.data }).toArray(i1 -> new NDArray[i1][]);
  }

}
