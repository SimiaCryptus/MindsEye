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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.util.io.KryoUtil;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by Andrew Charneski on 5/8/2017.
 */
public class ComponentTestUtil {
  /**
   * The constant deltaFactor.
   */
  public static final double deltaFactor = 1e-1;
  private static final Logger log = LoggerFactory.getLogger(ComponentTestUtil.class);
  /**
   * The constant tolerance.
   */
  public static double tolerance;

  /**
   * Instantiates a new Component test util.
   */
  public ComponentTestUtil() {
    tolerance = 1e-4;
  }

  /**
   * Get feedback gradient tensor [ ].
   *
   * @param component       the component
   * @param inputIndex      the input index
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the tensor [ ]
   */
  public static Tensor[] getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor gradientBuffer = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final NNResult[] copyInput = Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList data) {
        }
        
        @Override
        public boolean isAlive() {
          return false;
        }
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList data) {
          Assert.assertEquals(1, data.length());
          IntStream.range(0, data.length()).forEach(dataIndex -> {
            Assert.assertArrayEquals(inputPrototype[inputIndex].getDimensions(), data.get(dataIndex).getDimensions());
            for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
              gradientBuffer.set(new int[]{i, j_}, data.get(dataIndex).getData()[i]);
            }
          });
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
      final Tensor[] data = new Tensor[]{new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      component.eval(new NNLayer.NNExecutionContext() {}, copyInput).accumulate(new DeltaSet(), new TensorArray(data));
    }
    return new Tensor[]{ gradientBuffer };
  }
  
  private static Tensor getLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet buffer = new DeltaSet();
      final Tensor[] data = new Tensor[]{new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      component.eval(new NNLayer.NNExecutionContext() {},inputPrototype).accumulate(buffer, new TensorArray(data));
      final Delta deltaFlushBuffer = buffer.map.values().stream().filter(x -> x.target == stateArray).findFirst().get();
      for (int i = 0; i < stateLen; i++) {
        gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
      }
    }
    return gradient;
  }
  
  /**
   * Measure feedback gradient tensor.
   *
   * @param component       the component
   * @param inputIndex      the input index
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the tensor
   */
  public static Tensor measureFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    final Tensor baseOutput = component.eval(new NNLayer.NNExecutionContext() {
    }, inputPrototype).getData().get(0);
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, deltaFactor * 1);
      final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      final Tensor evalProbe = component.eval(new NNLayer.NNExecutionContext() {
      }, copyInput).getData().get(0);
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }
  
  /**
   * Measure learning gradient tensor.
   *
   * @param component       the component
   * @param layerNum        the layer num
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the tensor
   */
  public static Tensor measureLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    final Tensor baseOutput = component.eval(new NNLayer.NNExecutionContext() {
    }, inputPrototype).getData().get(0);
    for (int i = 0; i < stateLen; i++) {
      final NNLayer copy = KryoUtil.kryo().copy(component);
      copy.state().get(layerNum)[i] += deltaFactor;
      final Tensor evalProbe = copy.eval(new NNLayer.NNExecutionContext() {
      }, inputPrototype).getData().get(0);
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / deltaFactor);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return gradient;
  }
  
  /**
   * Test.
   *
   * @param component       the component
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @throws Throwable the throwable
   */
  public static void test(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    for (int i = 0; i < inputPrototype.length; i++) {
      testFeedback(component, i, outputPrototype, inputPrototype);
    }
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      testLearning(component, i, outputPrototype, inputPrototype);
    }
  }
  
  /**
   * Test feedback.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @throws Throwable the throwable
   */
  public static void testFeedback(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    final Tensor measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype)[0];
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], tolerance);
      } catch (final Throwable e) {
        log.debug(String.format("Error Comparing element %s in feedback", i1));
        log.debug(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
        log.debug(String.format("measured/actual: %s", measuredGradient));
        log.debug(String.format("implemented/expected: %s", implementedGradient));
        log.debug(String.format("error: %s", measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }
  
  /**
   * Test learning.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @throws Throwable the throwable
   */
  public static void testLearning(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    final Tensor measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
    for (int i1 = 0; i1 < measuredGradient.dim(); i1++) {
      try {
        Assert.assertEquals(measuredGradient.getData()[i1], implementedGradient.getData()[i1], tolerance);
      } catch (final Throwable e) {
        log.debug(String.format("Error Comparing element %s in learning", i1));
        log.debug(String.format("Component: %s", component));
        log.debug(String.format("Inputs: %s", Arrays.toString(inputPrototype)));
        log.debug(String.format("Outputs: %s", component.eval(new NNLayer.NNExecutionContext() {
        }, inputPrototype).getData().get(0)));
        log.debug(String.format("Measured Gradient: %s", measuredGradient));
        log.debug(String.format("Implemented Gradient: %s", implementedGradient));
        log.debug(String.format("%s", measuredGradient.minus(implementedGradient)));
        throw e;
      }
    }
  }
}
