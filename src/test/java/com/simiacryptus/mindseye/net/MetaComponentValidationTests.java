/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.net.NNLayer.ConstNNResult;
import com.simiacryptus.mindseye.net.meta.AvgMetaLayer;
import com.simiacryptus.mindseye.net.meta.CrossDotMetaLayer;
import com.simiacryptus.mindseye.net.meta.Sparse01MetaLayer;
import com.simiacryptus.mindseye.net.synapse.BiasLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.lang.KryoUtil;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

public class MetaComponentValidationTests {
  public static final double deltaFactor = 1e-6;
  
  private static final Logger log = LoggerFactory.getLogger(MetaComponentValidationTests.class);
  
  public static Tensor[][] getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) {
    final Tensor[][] gradients = IntStream.range(0, inputPrototype[inputIndex].length)
                                     .mapToObj(i -> IntStream.range(0, outputPrototype.length)
                                                        .mapToObj(j -> new Tensor(inputPrototype[inputIndex][i].dim(), outputPrototype[j].dim()))
                                                        .toArray(j -> new Tensor[j]))
                                     .toArray(i -> new Tensor[i][]);
    for (int outCoord = 0; outCoord < outputPrototype[0].dim(); outCoord++) {
      final int outputCoord = outCoord;
      for (int outItem = 0; outItem < outputPrototype.length; outItem++) {
        final int outputItem = outItem;
        final NNResult[] copyInput = Arrays.stream(inputPrototype).map(x -> new ConstNNResult(x)).toArray(i -> new NNResult[i]);
        copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
          @Override
          public void accumulate(final DeltaSet buffer, final Tensor[] data) {
            IntStream.range(0, data.length).forEach(inputItem -> {
              double[] dataItem = data[inputItem].getData();
              for (int inputCoord = 0; inputCoord < dataItem.length; inputCoord++) {
                gradients[inputItem][outputItem].set(new int[]{inputCoord, outputCoord}, dataItem[inputCoord]);
              }
            });
          }
          
          @Override
          public boolean isAlive() {
            return true;
          }
        };
        Tensor[] deltas = Arrays.stream(outputPrototype).map(x -> new Tensor(x.getDims())).toArray(i -> new Tensor[i]);
        deltas[outItem].set(outputCoord, 1);
        component.eval(copyInput).accumulate(new DeltaSet(), deltas);
      }
    }
    return gradients;
  }
  
  public static Tensor[][] measureFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) {
    final Tensor[][] gradients = IntStream.range(0, inputPrototype[inputIndex].length)
                                     .mapToObj(i -> IntStream.range(0, outputPrototype.length)
                                                        .mapToObj(j -> new Tensor(inputPrototype[inputIndex][i].dim(), outputPrototype[j].dim()))
                                                        .toArray(j -> new Tensor[j]))
                                     .toArray(i -> new Tensor[i][]);
    final Tensor[] baseOutput = component.eval(inputPrototype).data;
    for (int inputItem = 0; inputItem < inputPrototype[inputIndex].length; inputItem++) {
      for (int inputCoord = 0; inputCoord < inputPrototype[inputIndex][0].dim(); inputCoord++) {
        final Tensor[][] copyInput = Arrays.stream(inputPrototype)
                                         .map(a -> Arrays.stream(a).map(b -> b.copy()).toArray(ii -> new Tensor[ii])).toArray(ii -> new Tensor[ii][]);
        copyInput[inputIndex][inputItem].add(inputCoord, deltaFactor * 1);
        final Tensor[] probeOutput = component.eval(copyInput).data;
        for (int outputItem = 0; outputItem < probeOutput.length; outputItem++) {
          double[] deltaData = probeOutput[outputItem].minus(baseOutput[outputItem]).scale(1. / deltaFactor).getData();
          for (int outputCoord = 0; outputCoord < deltaData.length; outputCoord++) {
            gradients[inputItem][outputItem].set(new int[]{inputCoord, outputCoord}, deltaData[outputCoord]);
          }
        }
      }
    }
    return gradients;
  }
  
  private static Tensor[] getLearningGradient(final NNLayer component, final int layerNum, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor[] gradient = IntStream.range(0, outputPrototype.length).mapToObj(i -> new Tensor(stateLen, outputPrototype[0].dim())).toArray(i -> new Tensor[i]);
    for (int outCoord = 0; outCoord < outputPrototype[0].dim(); outCoord++) {
      final int outputCoord = outCoord;
      for (int outputItem = 0; outputItem < outputPrototype.length; outputItem++) {
        final DeltaSet buffer = new DeltaSet();
        NNResult eval = component.eval(inputPrototype);
        Tensor[] feedback = IntStream.range(0, outputPrototype.length).mapToObj(i -> new Tensor(outputPrototype[0].getDims())).toArray(i -> new Tensor[i]);
        feedback[outputItem].getData()[outputCoord] = 1;
        eval.accumulate(buffer, feedback);
        final DeltaBuffer delta = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
        for (int stateIdx = 0; stateIdx < stateLen; stateIdx++) {
          gradient[outputItem].set(new int[]{stateIdx, outputCoord}, delta.delta[stateIdx]);
        }
      }
    }
    return gradient;
  }
  
  public static Tensor[] measureLearningGradient(final NNLayer component, final int layerNum, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final Tensor[] gradient = IntStream.range(0, outputPrototype.length).mapToObj(i -> new Tensor(stateLen, outputPrototype[0].dim())).toArray(i -> new Tensor[i]);
    NNResult baseEval = component.eval(inputPrototype);
    for (int stateIdx = 0; stateIdx < stateLen; stateIdx++) {
      final NNLayer copy = KryoUtil.kryo().copy(component);
      copy.state().get(layerNum)[stateIdx] += deltaFactor;
      NNResult eval = copy.eval(inputPrototype);
      for (int outputItem = 0; outputItem < outputPrototype.length; outputItem++) {
        final Tensor delta = eval.data[outputItem].minus(baseEval.data[outputItem]).scale(1. / deltaFactor);
        for (int outputCoord = 0; outputCoord < delta.dim(); outputCoord++) {
          gradient[outputItem].set(new int[]{stateIdx, outputCoord}, delta.getData()[outputCoord]);
        }
      }
    }
    return gradient;
  }
  
  public static void test(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    test(5, component, outputPrototype, inputPrototype);
  }
  
  public static void test(int n, final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    test(component, replicate(outputPrototype, n), Arrays.stream(inputPrototype).map(x -> replicate(x, n)).toArray(i -> new Tensor[i][]));
  }
  
  public static Tensor[] replicate(final Tensor outputPrototype, int n) {
    return IntStream.range(0, n).mapToObj(i -> outputPrototype).toArray(i -> new Tensor[i]);
  }
  
  public static void test(final NNLayer component, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) throws Throwable {
    for (int i = 0; i < inputPrototype.length; i++) {
      testFeedback(component, i, outputPrototype, inputPrototype);
    }
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }
  
  public static void testFeedback(final NNLayer component, final int i, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) throws Throwable {
    final Tensor[][] measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final Tensor[][] implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
    for (int j = 0; j < measuredGradient.length; j++) {
      for (int k = 0; k < measuredGradient[j].length; k++) {
        for (int i1 = 0; i1 < measuredGradient[j][k].dim(); i1++) {
          try {
            Assert.assertEquals(measuredGradient[j][k].getData()[i1], implementedGradient[j][k].getData()[i1], 1e-4);
          } catch (final Throwable e) {
            log.debug(String.format("Error Comparing element %s,%s,%s (i=%s)", j, k, i1, i));
            log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
            log.debug(String.format("measured/actual: %s", Arrays.stream(measuredGradient).map(x -> Arrays.toString(x)).reduce((a, b) -> a + "; " + b).get()));
            log.debug(String.format("implemented/expected: %s", Arrays.stream(implementedGradient).map(x -> Arrays.toString(x)).reduce((a, b) -> a + "; " + b).get()));
            log.debug(String.format("error[%s][%s]: %s", j, k, measuredGradient[j][k].minus(implementedGradient[j][k])));
            throw e;
          }
        }
      }
    }
  }
  
  public static void testLearning(final NNLayer component, final int i, final Tensor[] outputPrototype, final Tensor[]... inputPrototype) throws Throwable {
    final Tensor[] measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final Tensor[] implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
    for (int k = 0; k < measuredGradient.length; k++) {
      for (int i1 = 0; i1 < measuredGradient[k].dim(); i1++) {
        try {
          Assert.assertEquals(measuredGradient[k].getData()[i1], implementedGradient[k].getData()[i1], 1e-4);
        } catch (final Throwable e) {
          log.debug(String.format("Error Comparing element %s", i1));
          log.debug(String.format("Component: %s\nInputs: %s", component, Arrays.toString(inputPrototype)));
          log.debug(String.format("%s", Arrays.toString(measuredGradient)));
          log.debug(String.format("%s", Arrays.toString(implementedGradient)));
          log.debug(String.format("%s", measuredGradient[k].minus(implementedGradient[k])));
          throw e;
        }
      }
    }
  }
  
  @Test
  public void testBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSparse01MetaLayer() throws Throwable {
    final NNLayer component = new Sparse01MetaLayer();
    Tensor[][] inputPrototype = Arrays.stream(new Tensor[][]{replicate(new Tensor(3), 5)})
                                    .map(x -> Arrays.stream(x).map(y -> y.map(z -> Util.R.get().nextDouble())).toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
    Tensor[] outputPrototype = replicate(new Tensor(3), 1);
    test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testCrossDotMetaLayer() throws Throwable {
    final NNLayer component = new CrossDotMetaLayer();
    Tensor[][] inputPrototype = Arrays.stream(new Tensor[][]{replicate(new Tensor(3), 5)})
                                    .map(x -> Arrays.stream(x).map(y -> y.map(z -> Util.R.get().nextDouble())).toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
    Tensor[] outputPrototype = replicate(new Tensor(3, 3), 1);
    test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testAvgMetaLayer() throws Throwable {
    final NNLayer component = new AvgMetaLayer();
    Tensor[][] inputPrototype = Arrays.stream(new Tensor[][]{replicate(new Tensor(3), 5)})
                                    .map(x -> Arrays.stream(x).map(y -> y.map(z -> Util.R.get().nextDouble())).toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
    Tensor[] outputPrototype = replicate(new Tensor(3), 1);
    test(component, outputPrototype, inputPrototype);
  }
  
  
}
