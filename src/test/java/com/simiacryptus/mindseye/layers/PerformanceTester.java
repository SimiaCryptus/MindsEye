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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.lang.TimedResult;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class PerformanceTester {
  
  private static final Logger log = LoggerFactory.getLogger(PerformanceTester.class);
  
  private boolean testLearning = true;
  private boolean testFeedback = true;
  private int samples = 100;
  
  /**
   * Instantiates a new Derivative tester.
   *
   */
  public PerformanceTester() {

  }
  
  /**
   * Test.
   *
   * @param component       the component
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  public void test(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) {
    if (isTestFeedback()) {
      DoubleStatistics statistics = IntStream.range(0, inputPrototype.length).mapToObj(i -> {
        return testFeedbackPerformance(component, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get();
      System.out.println(String.format("Forward performance: %.4f +- %.4f [%.4f - %.4f]",
        statistics.getAverage() * 1e4, statistics.getStandardDeviation() * 1e4, statistics.getMin() * 1e4, statistics.getMax() * 1e4));
    }
    if (isTestLearning()) {
      DoubleStatistics statistics = IntStream.range(0, component.state().size()).mapToObj(i -> {
        return testLearningPerformance(component, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).orElseGet(()->null);
      if(null != statistics) System.out.println(String.format("Backward performance: %.4f +- %.4f [%.4f - %.4f]",
        statistics.getAverage() * 1e4, statistics.getStandardDeviation() * 1e4, statistics.getMin() * 1e4, statistics.getMax() * 1e4));
    }
  }
  
  private Tensor getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
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
      final Tensor[] data = {new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
        (d, exe) -> {
          NNResult eval = component.eval(exe, copyInput);
          Tensor tensor = eval.getData().get(0);
          eval.accumulate(new DeltaSet(), new TensorArray(data));
          return tensor;
        }, (a, b) -> a.add(b));
    }
    return gradientBuffer;
  }
  
  private Tensor getLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet buffer = new DeltaSet();
      final Tensor[] data = {new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      
      GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
        (d, exe) -> {
          NNResult eval = component.eval(exe, NNResult.batchResultArray(d.toArray(new Tensor[][]{})));
          Tensor tensor = eval.getData().get(0);
          eval.accumulate(buffer, new TensorArray(data));
          return tensor;
        }, (a, b) -> a.add(b));
      
      
      final DoubleBuffer deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray).findFirst().get();
      for (int i = 0; i < stateLen; i++) {
        gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
      }
    }
    return gradient;
  }
  
  /**
   * Test feedback.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  protected DoubleStatistics testFeedbackPerformance(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    try {
      return new DoubleStatistics().accept(IntStream.range(0,samples).mapToLong(l->
        TimedResult.time(()->getFeedbackGradient(component, i, outputPrototype, inputPrototype)).timeNanos
      ).mapToDouble(x->x/1e9).toArray());
    } catch (final Throwable e) {
      System.out.println(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
      throw e;
    }
  }
  
  /**
   * Test learning.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  protected DoubleStatistics testLearningPerformance(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    try {
      return new DoubleStatistics().accept(IntStream.range(0,samples).mapToLong(l->
        TimedResult.time(()->getLearningGradient(component, i, outputPrototype, inputPrototype)).timeNanos
      ).mapToDouble(x->x/1e9).toArray());
  } catch (final Throwable e) {
      System.out.println(String.format("Component: %s", component));
      System.out.println(String.format("Inputs: %s", Arrays.toString(inputPrototype)));
      System.out.println(String.format("Outputs: %s", outputPrototype));
      throw e;
    }
    
  }
  
  /**
   * Is run learning boolean.
   *
   * @return the boolean
   */
  public boolean isTestLearning() {
    return testLearning;
  }
  
  /**
   * Sets run learning.
   *
   * @param testLearning the run learning
   * @return the run learning
   */
  public PerformanceTester setTestLearning(boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }
  
  /**
   * Is run feedback boolean.
   *
   * @return the boolean
   */
  public boolean isTestFeedback() {
    return testFeedback;
  }
  
  /**
   * Sets run feedback.
   *
   * @param testFeedback the run feedback
   * @return the run feedback
   */
  public PerformanceTester setTestFeedback(boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }
}
