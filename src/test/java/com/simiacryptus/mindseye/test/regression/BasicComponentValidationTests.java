package com.simiacryptus.mindseye.test.regression;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.net.reducers.ProductLayer;
import com.simiacryptus.mindseye.net.reducers.SumLayer;

public class BasicComponentValidationTests {
  public static final double deltaFactor = 1e-6;

  private static final Logger log = LoggerFactory.getLogger(BasicComponentValidationTests.class);

  public static NDArray getFeedbackGradient(final NNLayer<?> component, final int inputIndex, final NDArray outputPrototype, final NDArray... inputPrototype) {
    final NDArray gradient = new NDArray(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final NNResult[] copyInput = java.util.Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
        @Override
        public void accumulate(final DeltaSet buffer, final NDArray data) {
        }

        @Override
        public boolean isAlive() {
          return false;
        }
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
        @Override
        public void accumulate(final DeltaSet buffer, final NDArray data) {
          for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
            gradient.set(new int[] { i, j_ }, data.getData()[i]);
          }
        }

        @Override
        public boolean isAlive() {
          return true;
        }
      };
      component.eval(copyInput).accumulate(new DeltaSet(), new NDArray(outputPrototype.getDims()).set(j, 1));
    }
    return gradient;
  }

  private static NDArray getLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray outputPrototype, final NDArray... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet buffer = new DeltaSet();
      component.eval(inputPrototype).accumulate(buffer, new NDArray(outputPrototype.getDims()).set(j, 1));
      final DeltaBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
      for (int i = 0; i < stateLen; i++) {
        gradient.set(new int[] { i, j_ }, deltaFlushBuffer.getCalcVector()[i]);
      }
    }
    return gradient;
  }

  public static NDArray measureFeedbackGradient(final NNLayer<?> component, final int inputIndex, final NDArray outputPrototype, final NDArray... inputPrototype) {
    final NDArray measuredGradient = new NDArray(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    final NDArray baseOutput = component.eval(inputPrototype).data[0];
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      final NDArray inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, deltaFactor * 1);
      final NDArray[] copyInput = java.util.Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      final NDArray evalProbe = component.eval(copyInput).data[0];
      final NDArray delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[] { i, j }, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }

  public static NDArray measureLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray outputPrototype, final NDArray... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final NDArray gradient = new NDArray(stateLen, outputPrototype.dim());
    final NDArray baseOutput = component.eval(inputPrototype).data[0];
    for (int i = 0; i < stateLen; i++) {
      final NNLayer<?> copy = Util.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      final NDArray evalProbe = copy.eval(inputPrototype).data[0];
      final NDArray delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[] { i, j }, delta.getData()[j]);
      }
    }
    return gradient;
  }

  public static void test(final NNLayer<?> component, final NDArray outputPrototype, final NDArray... inputPrototype) throws Throwable {
    for (int i = 0; i < inputPrototype.length; i++) {
      testFeedback(component, i, outputPrototype, inputPrototype);
    }
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  public static void testFeedback(final NNLayer<?> component, final int i, final NDArray outputPrototype, final NDArray... inputPrototype) throws Throwable {
    final NDArray measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final NDArray implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
      } catch (final Throwable e) {
        log.debug(String.format("Error Comparing element %s", i1));
        log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, java.util.Arrays.toString(inputPrototype), outputPrototype));
        log.debug(String.format("%s", measuredGradient));
        log.debug(String.format("%s", implementedGradient));
        log.debug(String.format("%s", measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }

  public static void testLearning(final NNLayer<?> component, final int i, final NDArray outputPrototype, final NDArray... inputPrototype) throws Throwable {
    final NDArray measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final NDArray implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
      } catch (final Throwable e) {
        log.debug(String.format("Error Comparing element %s", i1));
        log.debug(String.format("Component: %s\nInputs: %s", component, java.util.Arrays.toString(inputPrototype)));
        log.debug(String.format("%s", measuredGradient));
        log.debug(String.format("%s", implementedGradient));
        log.debug(String.format("%s", measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }

  @org.junit.Test
  public void testBiasLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(3);
    final NDArray inputPrototype = new NDArray(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Throwable {
    final NDArray outputPrototype = new NDArray(2);
    final NDArray inputPrototype = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Throwable {
    final NDArray outputPrototype = new NDArray(2);
    final NDArray inputPrototype = new NDArray(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testEntropyLossLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(1);
    NDArray inputPrototype1 = new NDArray(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype1 = inputPrototype1.scale(1. / inputPrototype1.l1());
    NDArray inputPrototype2 = new NDArray(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype2 = inputPrototype2.scale(1. / inputPrototype2.l1());
    final NNLayer<?> component = new EntropyLossLayer();
    final NDArray[] inputPrototype = { inputPrototype1, inputPrototype2 };
    testFeedback(component, 0, outputPrototype, inputPrototype);
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  @org.junit.Test
  public void testProductLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(1);
    final NDArray inputPrototype1 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NDArray inputPrototype2 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ProductLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(3);
    final NDArray inputPrototype = new NDArray(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SigmoidActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSoftmaxLayer() throws Throwable {
    final NDArray inputPrototype = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NDArray outputPrototype = inputPrototype.copy();
    final NNLayer<?> component = new SoftmaxActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqActivationLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(3);
    final NDArray inputPrototype = new NDArray(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SqActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqLossLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(1);
    final NDArray inputPrototype1 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NDArray inputPrototype2 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SqLossLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumLayer() throws Throwable {
    final NDArray outputPrototype = new NDArray(1);
    final NDArray inputPrototype1 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NDArray inputPrototype2 = new NDArray(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

}
