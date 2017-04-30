package com.simiacryptus.mindseye.test.regression;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.activation.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.activation.MaxConstLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.EntropyLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;

public class MediaComponentValidationTests {
  public static final double deltaFactor = 1e-5;

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MediaComponentValidationTests.class);

  private static void test(final NNLayer<?> component, final Tensor outputPrototype, final Tensor inputPrototype) throws Throwable {
    BasicComponentValidationTests.test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ConvolutionSynapseLayer(new int[] { 2, 2 }, 1).addWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 2, 1);
    final Tensor inputPrototype = new Tensor(2, 3, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ConvolutionSynapseLayer(new int[] { 2, 2 }, 1).addWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer3() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(1, 1, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ConvolutionSynapseLayer(new int[] { 1, 1 }, 4).addWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer4() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ConvolutionSynapseLayer(new int[] { 2, 3 }, 4).addWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testL1NormalizationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new L1NormalizationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMaxConstLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new MaxConstLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMaxEntLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    final NNLayer<?> component = new DAGNetwork().add(new L1NormalizationLayer()).add(new EntropyLayer());
    test(component, outputPrototype, inputPrototype1);
  }

  @org.junit.Test
  public void testMaxSubsampleLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new MaxSubsampleLayer(2, 2, 1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSumSubsampleLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumSubsampleLayer(2, 2, 1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSumSubsampleLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumSubsampleLayer(3, 5, 1);
    test(component, outputPrototype, inputPrototype);
  }

}
