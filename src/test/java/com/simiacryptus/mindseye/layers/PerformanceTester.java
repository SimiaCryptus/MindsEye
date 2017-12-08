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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class PerformanceTester {
  
  private static final Logger log = LoggerFactory.getLogger(PerformanceTester.class);
  
  private boolean testLearning = true;
  private boolean testEvaluation = true;
  private final int samples = 10;
  
  /**
   * Instantiates a new Derivative tester.
   */
  public PerformanceTester() {
  
  }
  
  /**
   * Test.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void test(final NNLayer component, final Tensor... inputPrototype) {
    Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
    if (isTestEvaluation()) {
      DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testEvaluationPerformance(component, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get();
      System.out.println(String.format("Evaluation performance: %.4f +- %.4f [%.4f - %.4f]",
        statistics.getAverage() * 1e4, statistics.getStandardDeviation() * 1e4, statistics.getMin() * 1e4, statistics.getMax() * 1e4));
    }
    if (isTestLearning()) {
      DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testLearningPerformance(component, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).orElseGet(() -> null);
      if (null != statistics) {
        System.out.println(String.format("Learning performance: %.4f +- %.4f [%.4f - %.4f]",
          statistics.getAverage() * 1e4, statistics.getStandardDeviation() * 1e4, statistics.getMin() * 1e4, statistics.getMax() * 1e4));
      }
    }
  }
  
  
  /**
   * Test feedback.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the double statistics
   */
  protected DoubleStatistics testEvaluationPerformance(final NNLayer component, final Tensor... inputPrototype) {
    return new DoubleStatistics().accept(IntStream.range(0, samples).mapToLong(l ->
      TimedResult.time(() -> GpuController.call(exe -> {
        return component.eval(exe, inputPrototype);
      })).timeNanos
    ).mapToDouble(x -> x / 1e9).toArray());
  }
  
  /**
   * Test learning.
   *
   * @param component       the component
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the double statistics
   */
  protected DoubleStatistics testLearningPerformance(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) {
    return new DoubleStatistics().accept(IntStream.range(0, samples).mapToLong(l ->
      GpuController.call(exe -> {
        NNResult eval = component.eval(exe, inputPrototype);
        return TimedResult.time(() -> {
          DeltaSet buffer = new DeltaSet();
          eval.accumulate(buffer, new TensorArray(outputPrototype));
          return buffer;
        });
      }).timeNanos).mapToDouble(x -> x / 1e9).toArray());
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
  public boolean isTestEvaluation() {
    return testEvaluation;
  }
  
  /**
   * Sets run feedback.
   *
   * @param testEvaluation the run feedback
   * @return the run feedback
   */
  public PerformanceTester setTestEvaluation(boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this;
  }
}
