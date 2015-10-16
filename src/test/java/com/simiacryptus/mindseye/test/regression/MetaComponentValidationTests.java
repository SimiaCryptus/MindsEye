package com.simiacryptus.mindseye.test.regression;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.meta.Sparse01MetaLayer;

public class MetaComponentValidationTests {
  public static final double deltaFactor = 1e-6;

  private static final Logger log = LoggerFactory.getLogger(MetaComponentValidationTests.class);

  public static NDArray[][] getFeedbackGradient(final NNLayer<?> component, final int inputIndex, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final NDArray[][] gradients = java.util.stream.IntStream.range(0, inputPrototype[inputIndex].length)
        .mapToObj(i->java.util.stream.IntStream.range(0, outputPrototype.length)
            .mapToObj(j->new NDArray(inputPrototype[inputIndex][i].dim(), outputPrototype[j].dim()))
            .toArray(j->new NDArray[j]))
        .toArray(i->new NDArray[i][]);
    for (int jj = 0; jj < outputPrototype[0].dim(); jj++) { final int j = jj;
      for(int k=0;k<outputPrototype.length;k++) { final int _k=k;
        final NNResult[] copyInput = java.util.Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
          @Override
          public void accumulate(final DeltaSet buffer, final NDArray[] data) {
          }
  
          @Override
          public boolean isAlive() {
            return false;
          }
        }).toArray(i -> new NNResult[i]);
        copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
          @Override
          public void accumulate(final DeltaSet buffer, final NDArray[] data) {
            java.util.stream.IntStream.range(0, data.length).forEach(dataIndex->{
              for (int i = 0; i < inputPrototype[inputIndex][dataIndex].dim(); i++) {
                gradients[dataIndex][_k].set(new int[] { i, j }, data[dataIndex].getData()[i]);
              }
              
            });
          }
  
          @Override
          public boolean isAlive() {
            return true;
          }
        };
        component.eval(copyInput).accumulate(new DeltaSet(), new NDArray[]{new NDArray(outputPrototype[k].getDims()).fill((kk)->kk==j?1:0)});
      }
    }
    return gradients;
  }

  private static NDArray[] getLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final NDArray[] gradient = java.util.stream.IntStream.range(0, outputPrototype.length).mapToObj(i->new NDArray(stateLen, outputPrototype[0].dim())).toArray(i->new NDArray[i]);
    for (int jj = 0; jj < outputPrototype[0].dim(); jj++) { final int j = jj;
      for (int k = 0; k < outputPrototype.length; k++) {
        final DeltaSet buffer = new DeltaSet();
        NNResult eval = component.eval(inputPrototype);
        NDArray[] feedback = new NDArray[]{new NDArray(outputPrototype[0].getDims())};
        feedback[k].getData()[j] = 1;
        eval.accumulate(buffer, feedback);
        final DeltaBuffer deltaFlushBuffer = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
        for (int i = 0; i < stateLen; i++) {
          gradient[k].set(new int[] { i, j }, deltaFlushBuffer.getCalcVector()[i]);
        }
      }
    }
    return gradient;
  }

  public static NDArray[][] measureFeedbackGradient(final NNLayer<?> component, final int inputIndex, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final NDArray[][] measuredGradient = java.util.stream.IntStream.range(0, inputPrototype[inputIndex].length)
        .mapToObj(i->java.util.stream.IntStream.range(0, outputPrototype.length)
            .mapToObj(j->new NDArray(inputPrototype[inputIndex][i].dim(), outputPrototype[j].dim()))
            .toArray(j->new NDArray[j]))
        .toArray(i->new NDArray[i][]);
    final NDArray[] baseOutput = component.eval(inputPrototype).data;
    for (int i = 0; i < inputPrototype[inputIndex][0].dim(); i++) {
      final NDArray[] delta = new NDArray[outputPrototype.length];
      for(int k=0;k<inputPrototype[inputIndex].length;k++) {
        final NDArray[][] copyInput = java.util.Arrays.stream(inputPrototype).map(a->java.util.Arrays.stream(a).map(b->b.copy()).toArray(ii->new NDArray[ii])).toArray(ii->new NDArray[ii][]);
        copyInput[inputIndex][k].add(i, deltaFactor * 1);
        final NDArray[] evalProbe = component.eval(copyInput).data;
        for(int j=0;j<outputPrototype.length;j++) {
          delta[j] = evalProbe[j].minus(baseOutput[j]).scale(1. / deltaFactor);
          for (int l = 0; l < delta[j].dim(); l++) {
            measuredGradient[k][j].set(new int[] { i, l }, delta[j].getData()[l]);
          }
        }
      }
    }
    return measuredGradient;
  }

  public static NDArray[] measureLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final NDArray[] gradient = java.util.stream.IntStream.range(0, outputPrototype.length).mapToObj(i->new NDArray(stateLen, outputPrototype[0].dim())).toArray(i->new NDArray[i]);
    NNResult baseEval = component.eval(inputPrototype);
    for (int i = 0; i < stateLen; i++) {
      final NNLayer<?> copy = Util.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      NNResult eval = copy.eval(inputPrototype);
      for (int k = 0; k < outputPrototype.length; k++) {
        final NDArray evalProbe = eval.data[k];
        final NDArray baseOutput = baseEval.data[k];
        final NDArray delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
        for (int j = 0; j < delta.dim(); j++) {
          gradient[k].set(new int[] { i, j }, delta.getData()[j]);
        }
      }
    }
    return gradient;
  }

  public static void test(final NNLayer<?> component, final NDArray outputPrototype, final NDArray... inputPrototype) throws Throwable {
    test(5, component, outputPrototype, inputPrototype);
  }

  public static void test(int n, final NNLayer<?> component, final NDArray outputPrototype, final NDArray... inputPrototype) throws Throwable {
    test(component, replicate(outputPrototype, n), java.util.Arrays.stream(inputPrototype).map(x->replicate(x, n)).toArray(i->new NDArray[i][]));
  }

  public static NDArray[] replicate(final NDArray outputPrototype, int n) {
    return java.util.stream.IntStream.range(0, n).mapToObj(i->outputPrototype).toArray(i->new NDArray[i]);
  }

  public static void test(final NNLayer<?> component, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) throws Throwable {
    for (int i = 0; i < inputPrototype.length; i++) {
      testFeedback(component, i, outputPrototype, inputPrototype);
    }
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }

  public static void testFeedback(final NNLayer<?> component, final int i, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) throws Throwable {
    final NDArray[][] measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final NDArray[][] implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
    for (int j = 0; j < measuredGradient.length; j++) {
      for (int k = 0; k < measuredGradient[j].length; k++) {
        for (int i1 = 0; i1 < measuredGradient[j][k].dim(); i1++) {
          try {
            org.junit.Assert.assertEquals(measuredGradient[j][k].getData()[i1], implementedGradient[j][k].getData()[i1], 1e-4);
          } catch (final Throwable e) {
            log.debug(String.format("Error Comparing element %s", i1));
            log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, java.util.Arrays.toString(inputPrototype), outputPrototype));
            log.debug(String.format("measured/actual: %s", measuredGradient));
            log.debug(String.format("implemented/expected: %s", implementedGradient));
            log.debug(String.format("error: %s", measuredGradient[j][k].minus(implementedGradient[j][k])));
            throw e;
          }
        }
      }
    }
  }

  public static void testLearning(final NNLayer<?> component, final int i, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) throws Throwable {
    final NDArray[] measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final NDArray[] implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
    for (int k = 0; k < measuredGradient.length; k++) {
      for (int i1 = 0; i1 < measuredGradient[k].dim(); i1++) {
        try {
          org.junit.Assert.assertEquals(measuredGradient[k].getData()[i1], implementedGradient[k].getData()[i1], 1e-4);
        } catch (final Throwable e) {
          log.debug(String.format("Error Comparing element %s", i1));
          log.debug(String.format("Component: %s\nInputs: %s", component, java.util.Arrays.toString(inputPrototype)));
          log.debug(String.format("%s", measuredGradient));
          log.debug(String.format("%s", implementedGradient));
          log.debug(String.format("%s", measuredGradient[k].minus(implementedGradient[k])));
          throw e;
        }
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
  public void testSparse01MetaLayer() throws Throwable {
    final NNLayer<?> component = new Sparse01MetaLayer();
    NDArray[][] inputPrototype = java.util.Arrays.stream(new NDArray[][]{replicate(new NDArray(3), 5)})
        .map(x->java.util.Arrays.stream(x).map(y->y.fill(() -> Util.R.get().nextDouble())).toArray(i->new NDArray[i]))
        .toArray(i->new NDArray[i][]);
    NDArray[] outputPrototype = replicate(new NDArray(3), 1);
    test(component, outputPrototype, inputPrototype);
  }


}
