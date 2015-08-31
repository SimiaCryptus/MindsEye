package com.simiacryptus.mindseye.test.demo;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Ignore;
import org.junit.Test;

import com.simiacryptus.mindseye.layers.BiasLayer;
import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.test.dev.MNIST;
import com.simiacryptus.mindseye.test.dev.SimpleMNIST;
import com.simiacryptus.mindseye.training.PipelineNetwork;
import com.simiacryptus.mindseye.training.Trainer;
import com.simiacryptus.mindseye.util.LabeledObject;
import com.simiacryptus.mindseye.util.Util;

public class MNISTClassificationTests extends ClassificationTestBase {
  
  public MNISTClassificationTests() {
    super();
    drawBG = false;
  }
  
  @Override
  public Trainer buildTrainer(NDArray[][] samples, PipelineNetwork net) {
    return super.buildTrainer(samples, net).setVerbose(true);
  }

  @Override
  public PipelineNetwork buildNetwork() {
    final int[] inputSize = new int[] { 28, 28 };
    final int[] outSize = new int[] { 10 };
    final PipelineNetwork net = new PipelineNetwork()
        .add(new DenseSynapseLayer(NDArray.dim(inputSize), outSize))
        .add(new BiasLayer(outSize))
        .add(new SoftmaxActivationLayer());
    return net;
  }
  
  public double[] inputToXY(NDArray input, int classificationActual, int classificationExpected) {
    return new double[] { //
        ((classificationActual + Util.R.get().nextDouble()) * 6. - 3), //
        (classificationExpected + Util.R.get().nextDouble()) * 6. - 3 //
    };
  }
  
  public Stream<LabeledObject<NDArray>> getTraining(final List<LabeledObject<NDArray>> buffer) {
    return Util.shuffle(buffer, SimpleMNIST.random).parallelStream().limit(1000);
  }
  
  @Test // (expected = RuntimeException.class)
  public void test() throws Exception {
    int maxSize = 1000;
    test(Util.shuffle(MNIST.trainingDataStream().collect(Collectors.toList()), SimpleMNIST.random).parallelStream().limit(maxSize)
        .map(obj -> new NDArray[] { obj.data, SimpleMNIST.toOutNDArray(SimpleMNIST.toOut(obj.label), 10) })
        .toArray(i -> new NDArray[i][]));
  }

  @Override
  public void verify(Trainer trainer) {
    trainer.verifyConvergence(0, 0.0, 1);
  }
  
}