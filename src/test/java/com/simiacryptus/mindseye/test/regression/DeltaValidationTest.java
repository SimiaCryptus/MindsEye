package com.simiacryptus.mindseye.test.regression;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.DeltaFlushBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.basic.EntropyLossLayer;
import com.simiacryptus.mindseye.net.basic.ProductLayer;
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.basic.SqLossLayer;
import com.simiacryptus.mindseye.net.basic.SumLayer;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dev.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.dev.MinActivationLayer;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.dev.SqActivationLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxEntLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.util.Util;

public class DeltaValidationTest  {
  private static final Logger log = LoggerFactory.getLogger(DeltaValidationTest.class);
  
  public static final double deltaFactor = 1e-6;

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Throwable{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(()->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Throwable{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(()->Util.R.get().nextGaussian());
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
  public void testMaxSubsampleLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MaxSubsampleLayer(2,2,1);
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
  public void testBiasLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i->Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqLossLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray inputPrototype2 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SqLossLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testProductLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray inputPrototype2 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ProductLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray inputPrototype2 = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SumLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
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
  public void testEntropyLossLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(()->Util.R.get().nextDouble());
    inputPrototype1 = inputPrototype1.scale(1./inputPrototype1.l1());
    NDArray inputPrototype2 = new NDArray(2).fill(()->Util.R.get().nextDouble());
    inputPrototype2 = inputPrototype2.scale(1./inputPrototype2.l1());
    NNLayer<?> component = new EntropyLossLayer();
    NDArray[] inputPrototype = { inputPrototype1, inputPrototype2 };
    testFeedback(component, 0, outputPrototype, inputPrototype); 
    int layers = component.state().size();
    for(int i=0;i<layers;i++){
      testLearning(component, i, outputPrototype, inputPrototype); 
    }
  }

  @org.junit.Test
  public void testSqActivationLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SqActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMinActivationLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MinActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Throwable{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SigmoidActivationLayer();
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
  public void testSoftmaxLayer() throws Throwable{
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray outputPrototype = inputPrototype.copy();
    NNLayer<?> component = new SoftmaxActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }


  public static void test(NNLayer<?> component, NDArray outputPrototype, NDArray... inputPrototype) throws Throwable {
    for(int i=0;i<inputPrototype.length;i++){
      testFeedback(component, i, outputPrototype, inputPrototype); 
    }
    int layers = component.state().size();
    for(int i=0;i<layers;i++){
      testLearning(component, i, outputPrototype, inputPrototype); 
    }
  }

  public static void testLearning(NNLayer<?> component, int i, NDArray outputPrototype, NDArray... inputPrototype) throws Throwable {
    NDArray measuredGradient = measureLearningGradient(component, i, outputPrototype,inputPrototype);
    NDArray implementedGradient = getLearningGradient(component, i, outputPrototype,inputPrototype);
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
      } catch (Throwable e) {
        log.debug(String.format("Error Comparing element %s",i1));
        log.debug(String.format("Component: %s\nInputs: %s",component, java.util.Arrays.toString(inputPrototype)));
        log.debug(String.format("%s",measuredGradient));
        log.debug(String.format("%s",implementedGradient));
        log.debug(String.format("%s",measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }

  public static void testFeedback(NNLayer<?> component, int i, NDArray outputPrototype, NDArray... inputPrototype) throws Throwable {
    NDArray measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    NDArray implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
      } catch (Throwable e) {
        log.debug(String.format("Error Comparing element %s",i1));
        log.debug(String.format("Component: %s\nInputs: %s\noutput=%s",component, java.util.Arrays.toString(inputPrototype), outputPrototype));
        log.debug(String.format("%s",measuredGradient));
        log.debug(String.format("%s",implementedGradient));
        log.debug(String.format("%s",measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }



  public static NDArray measureFeedbackGradient(NNLayer<?> component, int inputIndex, NDArray outputPrototype, NDArray... inputPrototype) {
    NDArray measuredGradient = new NDArray(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    NDArray baseOutput = component.eval(inputPrototype).data;
    outputPrototype.set(baseOutput);
    for(int i=0;i<inputPrototype[inputIndex].dim();i++) {
      NDArray inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, deltaFactor * 1);
      NDArray[] copyInput = java.util.Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      NDArray evalProbe = component.eval(copyInput).data;
      NDArray delta = evalProbe.minus(baseOutput).scale(1./deltaFactor);
      for(int j=0;j<delta.dim();j++){
        measuredGradient.set(new int[]{i,j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }

  public static NDArray measureLearningGradient(NNLayer<?> component, int layerNum, NDArray outputPrototype, NDArray... inputPrototype) {
    int stateLen = component.state().get(layerNum).length;
    NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    NDArray baseOutput = component.eval(inputPrototype).data;
    for(int i=0;i<stateLen;i++) {
      NNLayer<?> copy = Util.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      NDArray evalProbe = copy.eval(inputPrototype).data;
      NDArray delta = evalProbe.minus(baseOutput).scale(1./deltaFactor);
      for(int j=0;j<delta.dim();j++){
        gradient.set(new int[]{i,j}, delta.getData()[j]);
      }
    }
    return gradient;
  }


  private static NDArray getLearningGradient(NNLayer<?> component, int layerNum, NDArray outputPrototype, NDArray... inputPrototype) {
    double[] stateArray = component.state().get(layerNum);
    int stateLen = stateArray.length;
    NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    for(int j=0;j<outputPrototype.dim();j++){
      int j_ = j;
      DeltaBuffer buffer = new DeltaBuffer();
      component.eval(inputPrototype).feedback(new NDArray(outputPrototype.getDims()).set(j, 1), buffer);
      DeltaFlushBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x->x.target==stateArray).findFirst().get();
      for(int i=0;i<stateLen;i++) {
        gradient.set(new int[]{i,j_}, deltaFlushBuffer.getCalcVector()[i]);
      }
    }
    return gradient;
  }

  public static NDArray getFeedbackGradient(NNLayer<?> component, int inputIndex, NDArray outputPrototype, NDArray... inputPrototype) {
    NDArray gradient = new NDArray(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      int j_ = j;
      NNResult[] copyInput = java.util.Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
        @Override
        public void feedback(NDArray data, DeltaBuffer buffer) {
        }

        @Override
        public boolean isAlive() {
          return false;
        }
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
        @Override
        public boolean isAlive() {
          return true;
        }

        @Override
        public void feedback(NDArray data, DeltaBuffer buffer) {
          for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
            gradient.set(new int[] { i, j_ }, data.getData()[i]);
          }
        }
      };
      component.eval(copyInput).feedback(new NDArray(outputPrototype.getDims()).set(j, 1), new DeltaBuffer());
    }
    return gradient;
  }

  
}
