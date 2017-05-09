package com.simiacryptus.mindseye.test.regression;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.lang.KryoUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Andrew Charneski on 5/8/2017.
 */
public class ComponentTestUtil {
    private static final Logger log = LoggerFactory.getLogger(ComponentTestUtil.class);
    
    public static final double deltaFactor = 1e-6;

    public static Tensor[] getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
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

    private static Tensor getLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
      final double[] stateArray = component.state().get(layerNum);
      final int stateLen = stateArray.length;
      final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
      for (int j = 0; j < outputPrototype.dim(); j++) {
        final int j_ = j;
        final DeltaSet buffer = new DeltaSet();
        component.eval(inputPrototype).accumulate(buffer, new Tensor[]{new Tensor(outputPrototype.getDims()).fill((k)->k==j_?1:0)});
        final DeltaBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
        for (int i = 0; i < stateLen; i++) {
            gradient.set(new int[] { i, j_ }, deltaFlushBuffer.delta[i]);
        }
      }
      return gradient;
    }

    public static Tensor measureFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
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

    public static Tensor measureLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
      final int stateLen = component.state().get(layerNum).length;
      final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
      final Tensor baseOutput = component.eval(inputPrototype).data[0];
      for (int i = 0; i < stateLen; i++) {
        final NNLayer copy = KryoUtil.kryo().copy(component);
        copy.state().get(layerNum)[i] += deltaFactor;
        final Tensor evalProbe = copy.eval(inputPrototype).data[0];
        final Tensor delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
        for (int j = 0; j < delta.dim(); j++) {
          gradient.set(new int[] { i, j }, delta.getData()[j]);
        }
      }
      return gradient;
    }

    public static void test(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
      for (int i = 0; i < inputPrototype.length; i++) {
        testFeedback(component, i, outputPrototype, inputPrototype);
      }
      final int layers = component.state().size();
      for (int i = 0; i < layers; i++) {
        testLearning(component, i, outputPrototype, inputPrototype);
      }
    }

    public static void testFeedback(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
      final Tensor measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
      final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype)[0];
      for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
        try {
          org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
        } catch (final Throwable e) {
          log.debug(String.format("Error Comparing element %s in feedback", i1));
          log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, java.util.Arrays.toString(inputPrototype), outputPrototype));
          log.debug(String.format("measured/actual: %s", measuredGradient));
          log.debug(String.format("implemented/expected: %s", implementedGradient));
          log.debug(String.format("error: %s", measuredGradient.minus(implementedGradient)));
          throw e;
        }
      }
    }

    public static void testLearning(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
      final Tensor measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
      final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
      for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
        try {
          org.junit.Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], 1e-4);
        } catch (final Throwable e) {
          log.debug(String.format("Error Comparing element %s in learning", i1));
          log.debug(String.format("Component: %s", component));
          log.debug(String.format("Inputs: %s", java.util.Arrays.toString(inputPrototype)));
          log.debug(String.format("Outputs: %s", component.eval(inputPrototype).data[0]));
          log.debug(String.format("Measured Gradient: %s", measuredGradient));
          log.debug(String.format("Implemented Gradient: %s", implementedGradient));
          log.debug(String.format("%s", measuredGradient.minus(implementedGradient)));
          throw e;
        }
      }
    }
}
