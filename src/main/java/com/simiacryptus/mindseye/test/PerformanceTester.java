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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.lang.TimedResult;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Performance tester.
 */
public class PerformanceTester {
  
  private int samples = 5;
  private int batches = 100;
  private boolean testLearning = true;
  private boolean testEvaluation = true;
  
  /**
   * Instantiates a new Performance tester.
   */
  public PerformanceTester() {
  
  }
  
  /**
   * Test.
   *  @param component      the component
   * @param inputPrototype the input prototype
   */
  public void test(final NNLayer component, final Tensor... inputPrototype) {
    System.out.println(String.format("%s batches", batches));
    System.out.println("Input Dimensions:");
    Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
    Arrays.stream(inputPrototype).map(t->"\t"+Arrays.toString(t.getDimensions())).forEach(System.out::println);
    System.out.println("Performance:");
    if (isTestEvaluation()) {
      DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testEvaluationPerformance(component, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get();
      System.out.println(String.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
        statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testLearningPerformance(component, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).orElseGet(() -> null);
      if (null != statistics) {
        System.out.println(String.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
          statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
      }
    }
  }
  
  
  /**
   * Test evaluation performance double statistics.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the double statistics
   */
  protected DoubleStatistics testEvaluationPerformance(final NNLayer component, final Tensor... inputPrototype) {
    DoubleStatistics statistics = new DoubleStatistics();
    statistics.accept(TimedResult.time(() -> GpuController.call(exe -> {
      Stream<Tensor[]> stream = IntStream.range(0, batches).mapToObj(x -> inputPrototype);
      Tensor[][] array = stream.toArray(i -> new Tensor[i][]);
      return component.eval(exe, NNResult.batchResultArray(array));
    })).timeNanos / 1e9);
    return statistics;
  }
  
  /**
   * Test learning performance double statistics.
   *
   * @param component       the component
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the double statistics
   */
  protected DoubleStatistics testLearningPerformance(final NNLayer component, final Tensor outputPrototype, final Tensor... inputPrototype) {
    DoubleStatistics statistics = new DoubleStatistics();
    TimedResult<DeltaSet> time = GpuController.call(exe -> {
      Tensor[][] data = IntStream.range(0, batches).mapToObj(x -> x).flatMap(x -> Stream.<Tensor[]>of(inputPrototype)).toArray(i -> new Tensor[i][]);
      NNResult result = component.eval(exe, NNResult.batchResultArray(data));
      DeltaSet buffer = new DeltaSet();
      return TimedResult.time(() -> {
        Tensor[] delta = result.getData().stream().map(x -> x.map(v -> 1.0)).toArray(i -> new Tensor[i]);
        result.accumulate(buffer, new TensorArray(delta));
        return buffer;
      });
    });
    statistics.accept(time.timeNanos / 1e9);
    return statistics;
  }
  
  /**
   * Is test learning boolean.
   *
   * @return the boolean
   */
  public boolean isTestLearning() {
    return testLearning;
  }
  
  /**
   * Sets test learning.
   *
   * @param testLearning the test learning
   * @return the test learning
   */
  public PerformanceTester setTestLearning(boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }
  
  /**
   * Is test evaluation boolean.
   *
   * @return the boolean
   */
  public boolean isTestEvaluation() {
    return testEvaluation;
  }
  
  /**
   * Sets test evaluation.
   *
   * @param testEvaluation the test evaluation
   * @return the test evaluation
   */
  public PerformanceTester setTestEvaluation(boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this;
  }
  
  /**
   * Gets samples.
   *
   * @return the samples
   */
  public int getSamples() {
    return samples;
  }
  
  /**
   * Sets samples.
   *
   * @param samples the samples
   * @return the samples
   */
  public PerformanceTester setSamples(int samples) {
    this.samples = samples;
    return this;
  }
  
  /**
   * Gets batches.
   *
   * @return the batches
   */
  public int getBatches() {
    return batches;
  }
  
  /**
   * Sets batches.
   *
   * @param batches the batches
   * @return the batches
   */
  public PerformanceTester setBatches(int batches) {
    this.batches = batches;
    return this;
  }
}
