package com.simiacryptus.mindseye.test.regression;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNLayer.ConstNNResult;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.meta.AvgMetaLayer;
import com.simiacryptus.mindseye.net.meta.CrossDotMetaLayer;
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
    for (int outCoord = 0; outCoord < outputPrototype[0].dim(); outCoord++) { final int outputCoord = outCoord;
      for(int outItem=0;outItem<outputPrototype.length;outItem++) { final int outputItem=outItem;
        final NNResult[] copyInput = java.util.Arrays.stream(inputPrototype).map(x -> new ConstNNResult(x)).toArray(i -> new NNResult[i]);
        copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
          @Override
          public void accumulate(final DeltaSet buffer, final NDArray[] data) {
            java.util.stream.IntStream.range(0, data.length).forEach(inputItem->{
              double[] dataItem = data[inputItem].getData();
              for (int inputCoord = 0; inputCoord < dataItem.length; inputCoord++) {
                gradients[inputItem][outputItem].set(new int[] { inputCoord, outputCoord }, dataItem[inputCoord]);
              }
            });
          }
  
          @Override
          public boolean isAlive() {
            return true;
          }
        };
        NDArray[] deltas = java.util.Arrays.stream(outputPrototype).map(x->new NDArray(x.getDims())).toArray(i -> new NDArray[i]);
        deltas[outItem].set(outputCoord,1);
        component.eval(copyInput).accumulate(new DeltaSet(), deltas);
      }
    }
    return gradients;
  }

  public static NDArray[][] measureFeedbackGradient(final NNLayer<?> component, final int inputIndex, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final NDArray[][] gradients = java.util.stream.IntStream.range(0, inputPrototype[inputIndex].length)
        .mapToObj(i->java.util.stream.IntStream.range(0, outputPrototype.length)
            .mapToObj(j->new NDArray(inputPrototype[inputIndex][i].dim(), outputPrototype[j].dim()))
            .toArray(j->new NDArray[j]))
        .toArray(i->new NDArray[i][]);
    final NDArray[] baseOutput = component.eval(inputPrototype).data;
    for(int inputItem=0;inputItem<inputPrototype[inputIndex].length;inputItem++) {
      for (int inputCoord = 0; inputCoord < inputPrototype[inputIndex][0].dim(); inputCoord++) {
        final NDArray[][] copyInput = java.util.Arrays.stream(inputPrototype)
            .map(a->java.util.Arrays.stream(a).map(b->b.copy()).toArray(ii->new NDArray[ii])).toArray(ii->new NDArray[ii][]);
        copyInput[inputIndex][inputItem].add(inputCoord, deltaFactor * 1);
        final NDArray[] probeOutput = component.eval(copyInput).data;
        for(int outputItem=0;outputItem<probeOutput.length;outputItem++) {
          double[] deltaData = probeOutput[outputItem].minus(baseOutput[outputItem]).scale(1. / deltaFactor).getData();
          for (int outputCoord = 0; outputCoord < deltaData.length; outputCoord++) {
            gradients[inputItem][outputItem].set(new int[] { inputCoord, outputCoord }, deltaData[outputCoord]);
          }
        }
      }
    }
    return gradients;
  }

  private static NDArray[] getLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final NDArray[] gradient = java.util.stream.IntStream.range(0, outputPrototype.length).mapToObj(i->new NDArray(stateLen, outputPrototype[0].dim())).toArray(i->new NDArray[i]);
    for (int outCoord = 0; outCoord < outputPrototype[0].dim(); outCoord++) { final int outputCoord = outCoord;
      for (int outputItem = 0; outputItem < outputPrototype.length; outputItem++) {
        final DeltaSet buffer = new DeltaSet();
        NNResult eval = component.eval(inputPrototype);
        NDArray[] feedback = java.util.stream.IntStream.range(0, outputPrototype.length).mapToObj(i->new NDArray(outputPrototype[0].getDims())).toArray(i->new NDArray[i]);
        feedback[outputItem].getData()[outputCoord] = 1;
        eval.accumulate(buffer, feedback);
        final DeltaBuffer delta = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
        for (int stateIdx = 0; stateIdx < stateLen; stateIdx++) {
          gradient[outputItem].set(new int[] { stateIdx, outputCoord }, delta.getCalcVector()[stateIdx]);
        }
      }
    }
    return gradient;
  }

  public static NDArray[] measureLearningGradient(final NNLayer<?> component, final int layerNum, final NDArray[] outputPrototype, final NDArray[]... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final NDArray[] gradient = java.util.stream.IntStream.range(0, outputPrototype.length).mapToObj(i->new NDArray(stateLen, outputPrototype[0].dim())).toArray(i->new NDArray[i]);
    NNResult baseEval = component.eval(inputPrototype);
    for (int stateIdx = 0; stateIdx < stateLen; stateIdx++) {
      final NNLayer<?> copy = Util.kryo().copy(component);
      copy.state().get(layerNum)[stateIdx] += deltaFactor;
      NNResult eval = copy.eval(inputPrototype);
      for (int outputItem = 0; outputItem < outputPrototype.length; outputItem++) {
        final NDArray delta = eval.data[outputItem].minus(baseEval.data[outputItem]).scale(1. / deltaFactor);
        for (int outputCoord = 0; outputCoord < delta.dim(); outputCoord++) {
          gradient[outputItem].set(new int[] { stateIdx, outputCoord }, delta.getData()[outputCoord]);
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
            log.debug(String.format("Error Comparing element %s,%s,%s (i=%s)",j,k,i1,i));
            log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, java.util.Arrays.toString(inputPrototype), outputPrototype));
            log.debug(String.format("measured/actual: %s", java.util.Arrays.stream(measuredGradient).map(x->java.util.Arrays.toString(x)).reduce((a,b)->a+"; "+b).get()));
            log.debug(String.format("implemented/expected: %s", java.util.Arrays.stream(implementedGradient).map(x->java.util.Arrays.toString(x)).reduce((a,b)->a+"; "+b).get()));
            log.debug(String.format("error[%s][%s]: %s", j, k, measuredGradient[j][k].minus(implementedGradient[j][k])));
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
        .map(x->java.util.Arrays.stream(x).map(y->y.map(z -> Util.R.get().nextDouble())).toArray(i->new NDArray[i])).toArray(i->new NDArray[i][]);
    NDArray[] outputPrototype = replicate(new NDArray(3), 1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testCrossDotMetaLayer() throws Throwable {
    final NNLayer<?> component = new CrossDotMetaLayer();
    NDArray[][] inputPrototype = java.util.Arrays.stream(new NDArray[][]{replicate(new NDArray(3), 5)})
        .map(x->java.util.Arrays.stream(x).map(y->y.map(z -> Util.R.get().nextDouble())).toArray(i->new NDArray[i])).toArray(i->new NDArray[i][]);
    NDArray[] outputPrototype = replicate(new NDArray(3,3), 1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testAvgMetaLayer() throws Throwable {
    final NNLayer<?> component = new AvgMetaLayer();
    NDArray[][] inputPrototype = java.util.Arrays.stream(new NDArray[][]{replicate(new NDArray(3), 5)})
        .map(x->java.util.Arrays.stream(x).map(y->y.map(z -> Util.R.get().nextDouble())).toArray(i->new NDArray[i])).toArray(i->new NDArray[i][]);
    NDArray[] outputPrototype = replicate(new NDArray(3), 1);
    test(component, outputPrototype, inputPrototype);
  }


}
