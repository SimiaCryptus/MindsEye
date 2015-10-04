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
import com.simiacryptus.mindseye.net.basic.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.basic.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.net.dev.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.dev.MinMaxFilterLayer;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.util.Util;

public class DeltaValidationTest  {
  private static final Logger log = LoggerFactory.getLogger(DeltaValidationTest.class);
  
  public static final double deltaFactor = 1e-6;

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Exception{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(()->Util.R.get().nextGaussian());
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Exception{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(()->Util.R.get().nextGaussian());
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testMinMaxLayer() throws Exception{
    NDArray outputPrototype = new NDArray(2);
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MinMaxFilterLayer();
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testMaxSubsampleLayer() throws Exception{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new MaxSubsampleLayer(2,2,1);
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testSumSubsampleLayer() throws Exception{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SumSubsampleLayer(2,2,1);
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testConvolutionSynapseLayer() throws Exception{
    NDArray outputPrototype = new NDArray(1,1,1);
    NDArray inputPrototype = new NDArray(2,2,1).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new ConvolutionSynapseLayer(new int[]{2,2},1);
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testBiasLayer() throws Exception{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i->Util.R.get().nextGaussian());
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Exception{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new SigmoidActivationLayer();
    test(outputPrototype, inputPrototype, component);
  }
  
  @org.junit.Test
  public void testL1NormalizationLayer() throws Exception{
    NDArray outputPrototype = new NDArray(3);
    NDArray inputPrototype = new NDArray(3).fill(()->Util.R.get().nextGaussian());
    NNLayer<?> component = new L1NormalizationLayer();
    test(outputPrototype, inputPrototype, component);
  }

  @org.junit.Test
  public void testSoftmaxLayer() throws Exception{
    NDArray inputPrototype = new NDArray(2).fill(()->Util.R.get().nextGaussian());
    NDArray outputPrototype = inputPrototype.copy();
    NNLayer<?> component = new SoftmaxActivationLayer();
    test(outputPrototype, inputPrototype, component);
  }


  public static void test(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) throws Exception {
    {
      NDArray measuredGradient = measureFeedbackGradient(outputPrototype, inputPrototype, component);
      NDArray implementedGradient = getFeedbackGradient(outputPrototype, inputPrototype, component);
      assertEquals(measuredGradient, implementedGradient);
    }
    int layers = component.state().size();
    for(int i=0;i<layers;i++){
      NDArray measuredGradient = measureLearningGradient(outputPrototype, inputPrototype, component,i);
      NDArray implementedGradient = getLearningGradient(outputPrototype, inputPrototype, component,i);
      assertEquals(measuredGradient, implementedGradient);
    }
  }



  public static void assertEquals(NDArray measuredGradient, NDArray implementedGradient) throws Exception {
    try {
      for (int i = 0; i < measuredGradient.dim(); i++) {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i], implementedGradient.getData()[i], 1e-4);
      } 
    } catch (Throwable e) {
      log.debug(String.format("%s",measuredGradient));
      log.debug(String.format("%s",implementedGradient));
      log.debug(String.format("%s",measuredGradient.minus(implementedGradient)));
      throw e;
    }
  }


  public static NDArray measureFeedbackGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) {
    NDArray measuredGradient = new NDArray(inputPrototype.dim(), outputPrototype.dim());
    NDArray baseOutput = component.eval(new EvaluationContext(), inputPrototype).data;
    for(int i=0;i<inputPrototype.dim();i++) {
      NDArray inputProbe = inputPrototype.copy();
      inputProbe.add(i, deltaFactor * 1);
      NDArray evalProbe = component.eval(new EvaluationContext(), inputProbe).data;
      NDArray delta = evalProbe.minus(baseOutput).scale(1./deltaFactor);
      for(int j=0;j<delta.dim();j++){
        measuredGradient.set(new int[]{i,j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }

  public static NDArray measureLearningGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component, int layerNum) {
    int stateLen = component.state().get(layerNum).length;
    NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    NDArray baseOutput = component.eval(new EvaluationContext(), inputPrototype).data;
    for(int i=0;i<stateLen;i++) {
      NNLayer<?> copy = Util.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      NDArray evalProbe = copy.eval(new EvaluationContext(), inputPrototype).data;
      NDArray delta = evalProbe.minus(baseOutput).scale(1./deltaFactor);
      for(int j=0;j<delta.dim();j++){
        gradient.set(new int[]{i,j}, delta.getData()[j]);
      }
    }
    return gradient;
  }


  private static NDArray getLearningGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component, int layerNum) {
    double[] stateArray = component.state().get(layerNum);
    int stateLen = stateArray.length;
    NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    for(int j=0;j<outputPrototype.dim();j++){
      int j_ = j;
      EvaluationContext evaluationContext = new EvaluationContext();
      DeltaBuffer buffer = new DeltaBuffer();
      component.eval(evaluationContext, inputPrototype).feedback(new NDArray(outputPrototype.getDims()).set(j, 1), buffer);
      DeltaFlushBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x->x.target==stateArray).findFirst().get();
      for(int i=0;i<stateLen;i++) {
        gradient.set(new int[]{i,j_}, deltaFlushBuffer.getCalcVector()[i]);
      }
    }
    return gradient;
  }

  public static NDArray getFeedbackGradient(NDArray outputPrototype, NDArray inputPrototype, NNLayer<?> component) {
    NDArray gradient = new NDArray(inputPrototype.dim(), outputPrototype.dim());
    for(int j=0;j<outputPrototype.dim();j++){
      int j_ = j;
      EvaluationContext evaluationContext = new EvaluationContext();
      component.eval(evaluationContext, new NNResult(evaluationContext, inputPrototype) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
        
        @Override
        public void feedback(NDArray data, DeltaBuffer buffer) {
          for(int i=0;i<inputPrototype.dim();i++) {
            gradient.set(new int[]{i,j_}, data.getData()[i]);
          }
        }
      }).feedback(new NDArray(outputPrototype.getDims()).set(j, 1), new DeltaBuffer());
    }
    return gradient;
  }

  
}
