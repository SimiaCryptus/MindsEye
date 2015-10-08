package com.simiacryptus.mindseye.test.regression;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SumLayer;
import com.simiacryptus.mindseye.net.dev.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.dev.L1SimpleNormalizationLayer;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.dev.ThresholdActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxConstLayer;
import com.simiacryptus.mindseye.net.media.MaxEntLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;

public class MediaComponentValidationTests  {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MediaComponentValidationTests.class);
  
  public static final double deltaFactor = 1e-6;

  private static void test(NNLayer<?> component, NDArray outputPrototype, NDArray inputPrototype) throws Throwable {
    BasicComponentValidationTests.test(component, outputPrototype, inputPrototype);
  }

  private static void test(NNLayer<?> component, NDArray outputPrototype, NDArray inputPrototype1, NDArray inputPrototype2) throws Throwable {
    BasicComponentValidationTests.test(component, outputPrototype, inputPrototype1,inputPrototype2);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer1() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ConvolutionSynapseLayer(new int[]{2,2},1).addWeights(()->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer2() throws Throwable{
    NDArray outputPrototype = new NDArray(1,2,1);
    NDArray inputPrototype = new NDArray(2,3,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ConvolutionSynapseLayer(new int[]{2,2},1).addWeights(()->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer3() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,2);
    NDArray inputPrototype = new NDArray(1,1,2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ConvolutionSynapseLayer(new int[]{1,1},4).addWeights(()->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer4() throws Throwable{
    NDArray outputPrototype = new NDArray(2,3,2);
    NDArray inputPrototype = new NDArray(3,5,2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ConvolutionSynapseLayer(new int[]{2,3},4).addWeights(()->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testL1NormalizationLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new L1NormalizationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testL1SimpleNormalizationLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new L1SimpleNormalizationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMaxEntLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray inputPrototype2 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DAGNetwork().add(new SoftmaxActivationLayer()).add(new MaxEntLayer()).add(new SumLayer());
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testMaxConstLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MaxConstLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMaxSubsampleLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MaxSubsampleLayer(2,2,1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMinActivationLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ThresholdActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }
  
  @org.junit.Test
  public void testMinMaxLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MinMaxFilterLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSumSubsampleLayer1() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SumSubsampleLayer(2,2,1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSumSubsampleLayer2() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,2);
    NDArray inputPrototype = new NDArray(3,5,2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SumSubsampleLayer(3,5,1);
    test(component, outputPrototype, inputPrototype);
  }

  
}
