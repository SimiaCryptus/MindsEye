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
import com.simiacryptus.util.io.KryoUtil;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class EquivalencyTester {
  
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);
  
  /**
   * The Probe size.
   */
  private final double tolerance;
  private boolean testLearning = true;
  private boolean testFeedback = true;
  
  /**
   * Instantiates a new Derivative tester.
   *
   * @param tolerance the tolerance
   */
  public EquivalencyTester(double tolerance) {
    this.tolerance = tolerance;
  }
  
  /**
   * Test.
   *
   * @param reference       the component
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  public ToleranceStatistics test(final NNLayer reference, final NNLayer subject, final Tensor outputPrototype, final Tensor... inputPrototype) {
    if(null == reference || null == subject) return new ToleranceStatistics();
    ToleranceStatistics statistics = new ToleranceStatistics();
    if (isTestFeedback()) {
      statistics = statistics.combine(IntStream.range(0, inputPrototype.length).mapToObj(i -> {
        return testFeedback(reference, subject, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get());
    }
    if (isTestLearning()) {
      statistics = statistics.combine(IntStream.range(0, inputPrototype.length).mapToObj(i -> {
        return testLearning(reference, subject, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get());
    }
    //System.out.println(String.format("Component: %s\nInputs: %s\noutput=%s", reference, Arrays.toString(inputPrototype), outputPrototype));
    System.out.println(String.format("Reference Layer Accuracy:"));
    System.out.println(String.format("absoluteTol: %s", statistics.absoluteTol.toString()));
    System.out.println(String.format("relativeTol: %s", statistics.relativeTol.toString()));
    return statistics;
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
   *  @param reference       the reference
   * @param subject
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  protected ToleranceStatistics testFeedback(final NNLayer reference, NNLayer subject, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor subjectGradient = getFeedbackGradient(subject, i, outputPrototype, inputPrototype);
    final Tensor referenceGradient = getFeedbackGradient(reference, i, outputPrototype, inputPrototype);
    try {
      ToleranceStatistics result = IntStream.range(0, subjectGradient.dim()).mapToObj(i1 -> {
        return new ToleranceStatistics().accumulate(subjectGradient.getData()[i1], referenceGradient.getData()[i1]);
      }).reduce((a, b) -> a.combine(b)).get();
      assert result.absoluteTol.getMax() < tolerance;
//      System.out.println(String.format("Output Dims: %s\nInputs: %s", Arrays.toString(outputPrototype.getDimensions()), Arrays.toString(inputPrototype)));
//      System.out.println(String.format("Subject Output: %s", subjectGradient));
//      System.out.println(String.format("Reference Output: %s", referenceGradient));
//      System.out.println(String.format("Error: %s", subjectGradient.minus(referenceGradient)));
//      System.out.println(String.format("Error Stats: %s", result));
      return result;
    } catch (final Throwable e) {
      System.out.println(String.format("Output Dims: %s\nInputs: %s", Arrays.toString(outputPrototype.getDimensions()), Arrays.toString(inputPrototype)));
      System.out.println(String.format("Subject Output: %s", subjectGradient));
      System.out.println(String.format("Reference Output: %s", referenceGradient));
      System.out.println(String.format("Error: %s", subjectGradient.minus(referenceGradient)));
      throw e;
    }
  }
  
  /**
   * Test learning.
   *  @param reference       the component
   * @param subject
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   */
  protected ToleranceStatistics testLearning(final NNLayer reference, NNLayer subject, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor subjectGradient = getLearningGradient(subject, i, outputPrototype, inputPrototype);
    final Tensor referenceGradient = getLearningGradient(reference, i, outputPrototype, inputPrototype);
  
    Tensor error = subjectGradient.minus(referenceGradient);
    try {
      ToleranceStatistics result = IntStream.range(0, subjectGradient.dim()).mapToObj(i1 -> {
        return new ToleranceStatistics().accumulate(subjectGradient.getData()[i1], referenceGradient.getData()[i1]);
      }).reduce((a, b) -> a.combine(b)).get();
      assert result.absoluteTol.getMax() < tolerance;
//      System.out.println(String.format("Subject Gradient: %s", subjectGradient));
//      System.out.println(String.format("Reference Gradient: %s", referenceGradient));
//      System.out.println(String.format("Error: %s", error));
//      System.out.println(String.format("Error Stats: %s", result));
      return result;
    } catch (final Throwable e) {
      System.out.println(String.format("Subject Gradient: %s", subjectGradient));
      System.out.println(String.format("Reference Gradient: %s", referenceGradient));
      System.out.println(String.format("Error: %s", error));
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
  public EquivalencyTester setTestLearning(boolean testLearning) {
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
  public EquivalencyTester setTestFeedback(boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }
}
