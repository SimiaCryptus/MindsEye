package com.simiacryptus.mindseye.test.demo.mnist;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.function.IntFunction;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.Coordinate;
import com.simiacryptus.mindseye.core.LabeledObject;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.TrainingContext;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.DAGNetwork.EvaluationContext;
import com.simiacryptus.mindseye.net.DAGNode;
import com.simiacryptus.mindseye.net.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
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

import groovy.lang.Tuple2;

public class MNISTAutoencoderTests {

  protected static final Logger log = LoggerFactory.getLogger(ClassificationTestBase.class);

  protected int getSampleSize(final Integer populationIndex, final int defaultNum) {
    return defaultNum;
  }

  public ReversibleNetwork buildNetwork() {
    List<int[]> sizes = new ArrayList<>();
    sizes.add(new int[] { 28, 28, 1 });
    sizes.add(new int[] { 100 });
    //sizes.add(new int[] { 100 });
    List<Tuple2<DenseSynapseLayerJBLAS, DenseSynapseLayerJBLAS>> codecs = new ArrayList<>();
    for(int i=1;i<sizes.size();i++) {
      codecs.add(createCodecPair(sizes.get(i-1), sizes.get(i)));
    }
    return stackedCodecNetwork(codecs);
  }

  public Tuple2<DenseSynapseLayerJBLAS, DenseSynapseLayerJBLAS> createCodecPair(final int[] outerSize, final int[] innerSize) {
    DenseSynapseLayerJBLAS encode = new DenseSynapseLayerJBLAS(NDArray.dim(outerSize), innerSize);//.setWeights(()->Util.R.get().nextGaussian()*0.1);
    DenseSynapseLayerJBLAS decode = new DenseSynapseLayerJBLAS(NDArray.dim(innerSize), outerSize).setWeights((Coordinate c)->{
      int[] traw = new int[]{c.coords[1],c.coords[0]};
      int tindex = encode.getWeights().index(traw);
      Coordinate transposed = new Coordinate(tindex, traw);
      double foo = encode.getWeights().get(transposed);
      return foo;
    });
    return new groovy.lang.Tuple2<>(encode, decode);
  }
  
  public static class ReversibleNetwork {
    DAGNetwork net;
    DAGNode center;
    DAGNode feedback;
    public DAGNode regularization;
    public ReversibleNetwork(DAGNetwork net, DAGNode center, DAGNode feedback) {
      super();
      this.net = net;
      this.center = center;
      this.feedback = feedback;
    }
    
  }

  private ReversibleNetwork stackedCodecNetwork(List<Tuple2<DenseSynapseLayerJBLAS, DenseSynapseLayerJBLAS>> codecs) {
    DAGNetwork net = new DAGNetwork();
    List<NNLayer<?>> weightNormalizationList = new ArrayList<>(); 
    for(int i=0;i<codecs.size();i++) {
      Tuple2<DenseSynapseLayerJBLAS, DenseSynapseLayerJBLAS> t = codecs.get(i);
      DenseSynapseLayerJBLAS encode = t.getFirst();
      net = net.add(encode);
      BiasLayer bias = new BiasLayer(encode.outputDims);
      weightNormalizationList.add(bias);
      net = net.add(bias);
      net = net.add(new SigmoidActivationLayer().setBalanced(false));
    }
    DAGNode center = net.getHead();
    for(int i=codecs.size()-1;i>=0;i--) {
      Tuple2<DenseSynapseLayerJBLAS, DenseSynapseLayerJBLAS> t = codecs.get(i);
      DenseSynapseLayerJBLAS decode = t.getSecond();
      net = net.add(decode);
      net = net.add(new BiasLayer(decode.outputDims));
      if(i>0){
        net = net.add(new SigmoidActivationLayer());
      }
    }
    ReversibleNetwork reversibleNetwork = new ReversibleNetwork(net, center, net.getHead());
    
    
    DAGNetwork _net = net;
    DAGNode[] norms = weightNormalizationList.stream().map(bias->{
      return _net.add(new WeightExtractor(0,bias), new DAGNode[]{}).add(new SqActivationLayer()).add(new SumReducerLayer()).getHead();
    }).toArray(i->new DAGNode[i]);
    reversibleNetwork.regularization = _net.add(new SumInputsLayer(), norms).getHead();
    //net = net.add(new SoftmaxActivationLayer());
    //net = net.add(new LinearActivationLayer().setWeight(300).freeze());
    //net = net.add(new LinearActivationLayer().setWeight(20*255).freeze());
    return reversibleNetwork;
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
    final NDArray[][] trainingData = transformDataSet(MNIST.trainingDataStream(), 100000, hash);
    final NDArray[][] validationData = transformDataSet(MNIST.validationDataStream(), 100, hash);
    final ReversibleNetwork net = buildNetwork();
    final List<String> report = new java.util.ArrayList<>();
    final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler = (trainedNetwork, trainingContext) -> {
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, java.util.Arrays.copyOf(trainingData, 100), net.feedback).stream().forEach(i->report.add(Util.toInlineImage(i, "TRAINING")));
      report.add("<hr/>");
      evaluateImageList(trainedNetwork, validationData, net.feedback).stream().forEach(i->report.add(Util.toInlineImage(i, "TEST")));
      report.add("<hr/>");
      return null;
    };
    try {
      {
        for(int i=0;i<10;i++) getTester(net, select(trainingData, 100), resultHandler).trainTo(.1);
        //getTester(net, select(trainingData, 1000), resultHandler).trainTo(.0);
//        getTester(net, java.util.Arrays.copyOf(trainingData, 40), resultHandler).verifyConvergence(1, 1);
//        getTester(net, java.util.Arrays.copyOf(trainingData, 50), resultHandler).verifyConvergence(.1, 1);
        //getTester(net, java.util.Arrays.copyOf(trainingData, 100), resultHandler).verifyConvergence(.1, 1);
//        getTester(net, trainingData, resultHandler).verifyConvergence(1, 1);
      }
    } finally {
      Util.report(report.stream().toArray(i -> new String[i]));
    }
  }

  private static NDArray[][] select(NDArray[][] array, int n) {
    return select(array,n,i->new NDArray[i][]);
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
    return java.util.Arrays.stream(array).sorted(java.util.Comparator.comparingInt(x->System.identityHashCode(x)^hash));
  }

  public Tester getTester(final ReversibleNetwork codec, NDArray[][] trainingData, final BiFunction<DAGNetwork, TrainingContext, Void> resultHandler) {
    Tester tester = new Tester();
    tester.setVerbose(true);
    GradientDescentTrainer trainer = tester.getGradientDescentTrainer();
    
    //supervisedNetwork.add(new SoftmaxActivationLayer());
    //supervisedNetwork.add(new LinearActivationLayer().setWeight(1./300).freeze());
    //expectedResult=expectedResult.add(new LinearActivationLayer().setWeight(1./(300)).freeze());
    //expectedResult = expectedResult.add(new SoftmaxActivationLayer());

    //NNLayer<?> loss = new EntropyLossLayer();
    NNLayer<?> loss = new SqLossLayer();
    DAGNode expectedResult = codec.net.getInput().get(1);
    DAGNode errResult = codec.net.add(loss, codec.feedback, expectedResult).add(new AvgMetaLayer()).getHead();
    DAGNode sparsityResult = codec.net
        .add(new Sparse01MetaLayer(), codec.center)
        .add(new SumReducerLayer())
        .add(new LinearActivationLayer().setWeight(0.1).freeze())
        .getHead();
    DAGNode regularizationResult = codec.net.add(new LinearActivationLayer().setWeight(0.1).freeze(), codec.regularization).getHead();
    codec.net.add(new VerboseWrapper("sums", new SumInputsLayer()), errResult, sparsityResult, regularizationResult);
    
    
    trainer.setNet(codec.net);
    trainer.setData(trainingData);
    if (null != resultHandler) {
      tester.handler.add(resultHandler);
    }
    tester.trainingContext().setTimeout(30, java.util.concurrent.TimeUnit.SECONDS);
    return tester;
  }

  public List<BufferedImage> evaluateImageList(DAGNetwork n, final NDArray[][] validationData, DAGNode feedback) {
    //final NNLayer<?> mainNetwork = n.getChild(feedback);
    return java.util.Arrays.stream(validationData)
        .map(x->feedback.get(n.buildExeCtx(x)).data[0])
        .map(x->Util.toImage(x))
        .collect(java.util.stream.Collectors.toList());
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
