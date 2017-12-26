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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Performance tester.
 */
public class PerformanceTester implements ComponentTest<ToleranceStatistics> {
  static final Logger logger = LoggerFactory.getLogger(PerformanceTester.class);
  
  private int batches = 100;
  private int samples = 5;
  private boolean testEvaluation = true;
  private boolean testLearning = true;
  
  /**
   * Instantiates a new Performance tester.
   */
  public PerformanceTester() {
  
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
  public PerformanceTester setBatches(final int batches) {
    this.batches = batches;
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
  public PerformanceTester setSamples(final int samples) {
    this.samples = samples;
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
  public PerformanceTester setTestEvaluation(final boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this;
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
  public PerformanceTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }
  
  /**
   * Test.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void test(final NNLayer component, final Tensor[] inputPrototype) {
    logger.info(String.format("%s batches", batches));
    logger.info("Input Dimensions:");
    final Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
    Arrays.stream(inputPrototype).map(t -> "\t" + Arrays.toString(t.getDimensions())).forEach(System.out::println);
    logger.info("Performance:");
    if (isTestEvaluation()) {
      final DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testEvaluationPerformance(component, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get();
      logger.info(String.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
        statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      final DoubleStatistics statistics = IntStream.range(0, samples).mapToObj(i -> {
        return testLearningPerformance(component, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).orElseGet(() -> null);
      if (null != statistics) {
        logger.info(String.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
          statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
      }
    }
  }
  
  /**
   * Test.
   *
   * @param log
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  @Override
  public ToleranceStatistics test(final NotebookOutput log, final NNLayer component, final Tensor... inputPrototype) {
    log.h1("Performance");
    if (component instanceof DAGNetwork) {
      TestUtil.instrumentPerformance(log, (DAGNetwork) component);
    }
    log.p("Now we execute larger-scale runs to benchmark performance:");
    log.code(() -> {
      test(component, inputPrototype);
    });
    if (component instanceof DAGNetwork) {
      TestUtil.extractPerformance(log, (DAGNetwork) component);
    }
    return null;
  }
  
  /**
   * Test evaluation performance double statistics.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the double statistics
   */
  protected DoubleStatistics testEvaluationPerformance(final NNLayer component, final Tensor... inputPrototype) {
    final DoubleStatistics statistics = new DoubleStatistics();
    statistics.accept(TimedResult.time(() -> GpuController.call(exe -> {
      final Stream<Tensor[]> stream = IntStream.range(0, batches).mapToObj(x -> inputPrototype);
      final Tensor[][] array = stream.toArray(i -> new Tensor[i][]);
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
    final DoubleStatistics statistics = new DoubleStatistics();
    final TimedResult<DeltaSet<NNLayer>> time = GpuController.call(exe -> {
      final Tensor[][] data = IntStream.range(0, batches).mapToObj(x -> x).flatMap(x -> Stream.<Tensor[]>of(inputPrototype)).toArray(i -> new Tensor[i][]);
      final NNResult result = component.eval(exe, NNResult.batchResultArray(data));
      final DeltaSet<NNLayer> buffer = new DeltaSet<NNLayer>();
      return TimedResult.time(() -> {
        final Tensor[] delta = result.getData().stream().map(x -> x.map(v -> 1.0)).toArray(i -> new Tensor[i]);
        result.accumulate(buffer, new TensorArray(delta));
        return buffer;
      });
    });
    statistics.accept(time.timeNanos / 1e9);
    return statistics;
  }
}
