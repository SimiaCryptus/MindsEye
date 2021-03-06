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

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.data.DoubleStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
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
  @Nonnull
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
  @Nonnull
  public PerformanceTester setSamples(final int samples) {
    this.samples = samples;
    return this;
  }

  /**
   * Is apply evaluation boolean.
   *
   * @return the boolean
   */
  public boolean isTestEvaluation() {
    return testEvaluation;
  }

  /**
   * Sets apply evaluation.
   *
   * @param testEvaluation the apply evaluation
   * @return the apply evaluation
   */
  @Nonnull
  public PerformanceTester setTestEvaluation(final boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this;
  }

  /**
   * Is apply learning boolean.
   *
   * @return the boolean
   */
  public boolean isTestLearning() {
    return testLearning;
  }

  /**
   * Sets apply learning.
   *
   * @param testLearning the apply learning
   * @return the apply learning
   */
  @Nonnull
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
  public void test(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    log.info(String.format("%s batch length, %s trials", batches, samples));
    log.info("Input Dimensions:");
    Arrays.stream(inputPrototype).map(t -> "\t" + Arrays.toString(t.getDimensions())).forEach(System.out::println);
    log.info("Performance:");
    List<Tuple2<Double, Double>> performance = IntStream.range(0, samples).mapToObj(i -> {
      return testPerformance(component, inputPrototype);
    }).collect(Collectors.toList());
    if (isTestEvaluation()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics().accept(performance.stream().mapToDouble(x -> x._1).toArray());
      log.info(String.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
          statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics().accept(performance.stream().mapToDouble(x -> x._2).toArray());
      if (null != statistics) {
        log.info(String.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]",
            statistics.getAverage(), statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
      }
    }
  }

  /**
   * Test.
   *  @param log
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, final Layer component, @Nonnull final Tensor... inputPrototype) {
    log.h1("Performance");
    if (component instanceof DAGNetwork) {
      TestUtil.instrumentPerformance((DAGNetwork) component);
    }
    log.p("Now we execute larger-scale runs to benchmark performance:");
    log.run(() -> {
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
  @Nonnull
  protected Tuple2<Double, Double> testPerformance(@Nonnull final Layer component, final Tensor... inputPrototype) {
    final Tensor[][] data = IntStream.range(0, batches).mapToObj(x -> x).flatMap(x -> Stream.<Tensor[]>of(inputPrototype)).toArray(i -> new Tensor[i][]);
    @Nonnull TimedResult<Result> timedEval = TimedResult.time(() -> {
      Result[] input = ConstantResult.batchResultArray(data);
      @Nullable Result result;
      try {
        result = component.eval(input);
      } finally {
        for (@Nonnull Result nnResult : input) {
          nnResult.freeRef();
          nnResult.getData().freeRef();
        }
      }
      return result;
    });
    final Result result = timedEval.result;
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    try {
      long timedBackprop = TimedResult.time(() -> {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(result.getData().stream().map(x -> {
          return x.mapAndFree(v -> 1.0);
        }).toArray(i -> new Tensor[i]));
        result.accumulate(buffer, tensorArray);
        assert tensorArray.currentRefCount() == 0;
        return buffer;
      }).timeNanos;
      return new Tuple2<>(timedEval.timeNanos / 1e9, timedBackprop / 1e9);
    } finally {
      buffer.freeRef();
      result.freeRef();
      result.getData().freeRef();
    }
  }

  @Nonnull
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
