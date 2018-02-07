/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.lang.Tuple2;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Performance tester.
 */
public class PerformanceTester extends ComponentTestBase<ToleranceStatistics> {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(PerformanceTester.class);
  
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
  @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
  public PerformanceTester setSamples(final int samples) {
    this.samples = samples;
    return this;
  }
  
  /**
   * Is run evaluation boolean.
   *
   * @return the boolean
   */
  public boolean isTestEvaluation() {
    return testEvaluation;
  }
  
  /**
   * Sets run evaluation.
   *
   * @param testEvaluation the run evaluation
   * @return the run evaluation
   */
  @javax.annotation.Nonnull
  public PerformanceTester setTestEvaluation(final boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this;
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
  @javax.annotation.Nonnull
  public ComponentTest<ToleranceStatistics> setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }
  
  /**
   * Test.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void test(@javax.annotation.Nonnull final NNLayer component, @javax.annotation.Nonnull final Tensor[] inputPrototype) {
    log.info(String.format("%s batch length, %s trials", batches, samples));
    log.info("Input Dimensions:");
    Arrays.stream(inputPrototype).map(t -> "\t" + Arrays.toString(t.getDimensions())).forEach(System.out::println);
    log.info("Performance:");
    List<Tuple2<Double, Double>> performance = IntStream.range(0, samples).mapToObj(i -> {
      return testPerformance(component, inputPrototype);
    }).collect(Collectors.toList());
    if (isTestEvaluation()) {
      @javax.annotation.Nonnull final DoubleStatistics statistics = new DoubleStatistics().accept(performance.stream().mapToDouble(x -> x._1).toArray());
      log.info(String.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
                             statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      @javax.annotation.Nonnull final DoubleStatistics statistics = new DoubleStatistics().accept(performance.stream().mapToDouble(x -> x._2).toArray());
      if (null != statistics) {
        log.info(String.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
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
  public @Nullable ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput log, final NNLayer component, @javax.annotation.Nonnull final Tensor... inputPrototype) {
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
   * Test learning performance double statistics.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the double statistics
   */
  @javax.annotation.Nonnull
  protected Tuple2<Double, Double> testPerformance(@javax.annotation.Nonnull final NNLayer component, final Tensor... inputPrototype) {
    final Tensor[][] data = IntStream.range(0, batches).mapToObj(x -> x).flatMap(x -> Stream.<Tensor[]>of(inputPrototype)).toArray(i -> new Tensor[i][]);
    @javax.annotation.Nonnull TimedResult<NNResult> timedEval = TimedResult.time(() -> {
      NNResult[] input = NNConstant.batchResultArray(data);
      NNResult result = component.eval(input);
      for (@javax.annotation.Nonnull NNResult nnResult : input) {
        nnResult.freeRef();
        nnResult.getData().freeRef();
      }
      return result;
    });
    final NNResult result = timedEval.result;
    @javax.annotation.Nonnull final DeltaSet<NNLayer> buffer = new DeltaSet<NNLayer>();
    long timedBackprop = TimedResult.time(() -> {
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(result.getData().stream().map(x -> x.map(v -> 1.0)).toArray(i -> new Tensor[i]));
      result.accumulate(buffer, tensorArray);
      tensorArray.freeRef();
      return buffer;
    }).timeNanos;
    buffer.freeRef();
    result.freeRef();
    result.getData().freeRef();
    return new Tuple2<>(timedEval.timeNanos / 1e9, timedBackprop / 1e9);
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "PerformanceTester{" +
      "batches=" + batches +
      ", samples=" + samples +
      ", testEvaluation=" + testEvaluation +
      ", testLearning=" + testLearning +
      '}';
  }
}
