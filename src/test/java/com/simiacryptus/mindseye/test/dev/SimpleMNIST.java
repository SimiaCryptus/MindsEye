package com.simiacryptus.mindseye.test.dev;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.data.LabeledObject;
import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;

public class SimpleMNIST {
  private static final Logger log = LoggerFactory.getLogger(SimpleMNIST.class);
  
  public static final Random random = new Random();
  
  @Test
  public void test() throws Exception {
    log.info("Starting");
    PipelineNetwork net = getNetwork();
    final List<LabeledObject<NDArray>> buffer = MNIST.trainingDataStream().collect(Collectors.toList());
    
    NDArray[][] data = getTraining(buffer)
        .map(o -> new NDArray[] { o.data, toOutNDArray(toOut(o.label), 10) })
        .toArray(i2 -> new NDArray[i2][]);
    
    getTrainer(net, data).verifyConvergence(10000, 0.01, 1);

    double prevRms = getVerification(buffer).mapToDouble(o1 -> net.eval(o1.data).errMisclassification(toOut(o1.label))).average().getAsDouble();
    log.info("Tested RMS Error: {}", prevRms);
    MNIST.report(net);
  }

  public Stream<LabeledObject<NDArray>> getVerification(final List<LabeledObject<NDArray>> buffer) {
    return buffer.parallelStream().limit(100);
  }

  public Stream<LabeledObject<NDArray>> getTraining(final List<LabeledObject<NDArray>> buffer) {
    return Util.shuffle(buffer, random).parallelStream().limit(1000);
  }

  public Trainer getTrainer(PipelineNetwork net, NDArray[][] data) {
    return net.trainer(data)
        .setDynamicRate(0.001)
        .setImprovementStaleThreshold(5)
        .setStaticRate(0.01)
        .setMutationAmount(0.2)
        .setVerbose(verbose>0);
  }
  
  final NDArray inputSize = new NDArray(28, 28);
  int verbose = 1;
  
  protected PipelineNetwork getNetwork() {
    PipelineNetwork net = new PipelineNetwork();
    net.add(new DenseSynapseLayer(net.eval(inputSize).data.dim(), new int[] { 10 }).setVerbose(verbose>1));
    net.add(new BiasLayer(net.eval(inputSize).data.getDims()).setVerbose(verbose>1));
    //net.add(new SigmoidActivationLayer().setVerbose(verbose));
    //net.add(new SoftmaxActivationLayer().setVerbose(verbose));
    return net;
  }
  
  public static NDArray toOutNDArray(int out, int max) {
    NDArray ndArray = new NDArray(max);
    ndArray.set(out, 1);
    return ndArray;
  }
  
  public static int toOut(final String label) {
    for (int i = 0; i < 10; i++)
    {
      if (label.equals("[" + i + "]")) return i;
    }
    throw new RuntimeException();
  }
  
}
