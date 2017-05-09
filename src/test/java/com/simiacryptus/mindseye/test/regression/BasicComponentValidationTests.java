package com.simiacryptus.mindseye.test.regression;

import com.simiacryptus.mindseye.net.activation.*;
import com.simiacryptus.mindseye.net.dev.ToeplitzSynapseLayerJBLAS;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerGPU;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.net.reducers.ProductLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.reducers.SumReducerLayer;

public class BasicComponentValidationTests {

  private static final Logger log = LoggerFactory.getLogger(BasicComponentValidationTests.class);

  @org.junit.Test
  public void testBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testToeplitzSynapseLayerJBLAS1() throws Throwable {
    final Tensor inputPrototype = new Tensor(3,3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3,3);
    final NNLayer component = new ToeplitzSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testToeplitzSynapseLayerJBLAS2() throws Throwable {
    final Tensor inputPrototype = new Tensor(3,3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3,3,2,3);
    final NNLayer component = new ToeplitzSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerJBLAS1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerGPU1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayerGPU(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testEntropyLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype1 = inputPrototype1.scale(1. / inputPrototype1.l1());
    Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype2 = inputPrototype2.scale(1. / inputPrototype2.l1());
    final NNLayer component = new EntropyLossLayer();
    final Tensor[] inputPrototype = { inputPrototype1, inputPrototype2 };
    ComponentTestUtil.testFeedback(component, 0, outputPrototype, inputPrototype);
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      ComponentTestUtil.testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  @org.junit.Test
  public void testProductLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ProductLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testReLu() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ReLuActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testLinear() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new LinearActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SigmoidActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSoftmaxLayer() throws Throwable {
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = inputPrototype.copy();
    final NNLayer component = new SoftmaxActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SqActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MeanSqLossLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumInputsLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumReducerLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

}
