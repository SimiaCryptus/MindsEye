package com.simiacryptus.mindseye.test.regression;

import com.simiacryptus.util.lang.KryoUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerGPU;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.net.loss.SqLossLayer;
import com.simiacryptus.mindseye.net.reducers.ProductLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.reducers.SumReducerLayer;

public class BasicComponentValidationTests {
  public static final double deltaFactor = 1e-6;

  private static final Logger log = LoggerFactory.getLogger(BasicComponentValidationTests.class);

  public static Tensor[] getFeedbackGradient(final NNLayer<?> component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor gradientA[] = java.util.stream.IntStream.range(0, 1)
        .mapToObj(i->new Tensor(inputPrototype[i].dim(), outputPrototype.dim()))
        .toArray(i->new Tensor[i]);
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final NNResult[] copyInput = java.util.Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
        @Override
        public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        }

        @Override
        public boolean isAlive() {
          return false;
        }
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
        @Override
        public void accumulate(final DeltaSet buffer, final Tensor[] data) {
          java.util.stream.IntStream.range(0, data.length).forEach(dataIndex->{
            for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
              gradientA[dataIndex].set(new int[] { i, j_ }, data[dataIndex].getData()[i]);
            }
            
          });
        }

        @Override
        public boolean isAlive() {
          return true;
        }
      };
      component.eval(copyInput).accumulate(new DeltaSet(), new Tensor[]{new Tensor(outputPrototype.getDims()).fill((k)->k==j_?1:0)});
    }
    return gradientA;
  }

  private static Tensor getLearningGradient(final NNLayer<?> component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet buffer = new DeltaSet();
      component.eval(inputPrototype).accumulate(buffer, new Tensor[]{new Tensor(outputPrototype.getDims()).fill((k)->k==j_?1:0)});
      final DeltaBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
      for (int i = 0; i < stateLen; i++) {
        gradient.set(new int[] { i, j_ }, deltaFlushBuffer.getCalcVector()[i]);
      }
    }
    return gradient;
  }

  public static Tensor measureFeedbackGradient(final NNLayer<?> component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    final Tensor baseOutput = component.eval(inputPrototype).data[0];
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, deltaFactor * 1);
      final Tensor[] copyInput = java.util.Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      final Tensor evalProbe = component.eval(copyInput).data[0];
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[] { i, j }, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }

  public static Tensor measureLearningGradient(final NNLayer<?> component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    final Tensor baseOutput = component.eval(inputPrototype).data[0];
    for (int i = 0; i < stateLen; i++) {
      final NNLayer<?> copy = KryoUtil.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      final Tensor evalProbe = copy.eval(inputPrototype).data[0];
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[] { i, j }, delta.getData()[j]);
      }
    }
    return gradient;
  }

  public static void test(final NNLayer<?> component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    for (int i = 0; i < inputPrototype.length; i++) {
      testFeedback(component, i, outputPrototype, inputPrototype);
    }
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  public static void testFeedback(final NNLayer<?> component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    final Tensor measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype)[0];
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
      } catch (final Throwable e) {
        log.debug(String.format("Error Comparing element %s", i1));
        log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, java.util.Arrays.toString(inputPrototype), outputPrototype));
        log.debug(String.format("measured/actual: %s", measuredGradient));
        log.debug(String.format("implemented/expected: %s", implementedGradient));
        log.debug(String.format("error: %s", measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }

  public static void testLearning(final NNLayer<?> component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    final Tensor measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
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
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerJBLAS1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayerJBLAS(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerGPU1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayerGPU(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testEntropyLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype1 = inputPrototype1.scale(1. / inputPrototype1.l1());
    Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype2 = inputPrototype2.scale(1. / inputPrototype2.l1());
    final NNLayer<?> component = new EntropyLossLayer();
    final Tensor[] inputPrototype = { inputPrototype1, inputPrototype2 };
    testFeedback(component, 0, outputPrototype, inputPrototype);
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  @org.junit.Test
  public void testProductLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ProductLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SigmoidActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSoftmaxLayer() throws Throwable {
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = inputPrototype.copy();
    final NNLayer<?> component = new SoftmaxActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SqActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SqLossLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumInputsLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumReducerLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

}
