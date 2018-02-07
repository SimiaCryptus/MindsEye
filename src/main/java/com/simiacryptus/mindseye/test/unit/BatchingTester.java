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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.SimpleListEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.data.DensityTree;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Batching tester.
 */
public class BatchingTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(BatchingTester.class);
  
  private final double tolerance;
  private int batchSize = 10;
  
  /**
   * Instantiates a new Batching tester.
   *
   * @param tolerance the tolerance
   */
  public BatchingTester(final double tolerance) {
    this.tolerance = tolerance;
  }
  
  /**
   * Gets randomize.
   *
   * @return the randomize
   */
  public double getRandom() {
    return 5 * (Math.random() - 0.5);
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param reference      the reference
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics test(final @Nullable NNLayer reference, final @NotNull Tensor[] inputPrototype) {
    if (null == reference) return new ToleranceStatistics();
  
    final TensorList[] inputTensorLists = Arrays.stream(inputPrototype).map(t ->
                                                                              TensorArray.wrap(IntStream.range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom()))
                                                                                                        .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
    final SimpleResult asABatch = SimpleListEval.run(reference, inputTensorLists);
    final List<SimpleEval> oneAtATime = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
                                                                                      Tensor[] inputTensors = IntStream.range(0, inputTensorLists.length)
                                                                                                                       .mapToObj(i -> inputTensorLists[i].get(batch)).toArray(i -> new Tensor[i]);
                                                                                      SimpleEval eval = SimpleEval.run(reference, inputTensors);
                                                                                      for (@NotNull Tensor tensor : inputTensors) {
                                                                                        tensor.freeRef();
                                                                                      }
                                                                                      return eval;
                                                                                    }
                                                                                   ).collect(Collectors.toList());
    for (@NotNull TensorList tensorList : inputTensorLists) {
      tensorList.freeRef();
    }
  
    final @NotNull ToleranceStatistics outputAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
                                                                                              Tensor batchTensor = asABatch.getOutput().get(batch);
                                                                                              ToleranceStatistics accumulate = new ToleranceStatistics().accumulate(
                                                                                                batchTensor.getData(),
                                                                                                oneAtATime.get(batch).getOutput().getData());
                                                                                              batchTensor.freeRef();
                                                                                              return accumulate;
                                                                                            }
                                                                                                    ).reduce((a, b) -> a.combine(b)).get();
    if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
      logger.info("Batch Output: " + asABatch.getOutput().stream().map(x -> x.prettyPrint()).collect(Collectors.toList()));
      logger.info("Singular Output: " + oneAtATime.stream().map(x -> x.getOutput().prettyPrint()).collect(Collectors.toList()));
      throw new AssertionError("Output Corrupt: " + outputAgreement);
    }
  
    final @NotNull ToleranceStatistics derivativeAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
      @NotNull IntFunction<ToleranceStatistics> statisticsFunction = input -> {
        Tensor a = asABatch.getDerivative()[input].get(batch);
        Tensor b = oneAtATime.get(batch).getDerivative()[input];
        @NotNull Tensor diff = a.minus(b);
        logger.info("Error: " + diff.prettyPrint());
        logger.info("Scalar Statistics: " + new ScalarStatistics().add(diff.getData()).getMetrics());
        double[][] points = Arrays.stream(diff.getData()).mapToObj(x -> new double[]{x}).toArray(i -> new double[i][]);
        logger.info("Density: " + new DensityTree("x").setMinSplitFract(1e-8).setSplitSizeThreshold(2).new Node(points));
        diff.freeRef();
        ToleranceStatistics toleranceStatistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
        a.freeRef();
        return toleranceStatistics;
      };
      return IntStream.range(0, inputPrototype.length).mapToObj(statisticsFunction).reduce((a, b) -> a.combine(b)).get();
    }).reduce((a, b) -> a.combine(b)).get();
  
    asABatch.freeRef();
    oneAtATime.forEach(x -> x.freeRef());
    if (!(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
      throw new AssertionError("Derivatives Corrupt: " + derivativeAgreement);
    }
    return derivativeAgreement.combine(outputAgreement);
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param log
   * @param reference      the reference
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(final @NotNull NotebookOutput log, final NNLayer reference, final @NotNull Tensor... inputPrototype) {
    log.h1("Batch Execution");
    log.p("Most layers, including this one, should behave the same no matter how the items are split between batches. We verify this:");
    return log.code(() -> {
      return test(reference, inputPrototype);
    });
  }
  
  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }
  
  /**
   * Sets batch size.
   *
   * @param batchSize the batch size
   * @return the batch size
   */
  public @NotNull BatchingTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
  
  @Override
  public @NotNull String toString() {
    return "BatchingTester{" +
      "tolerance=" + tolerance +
      ", batchSize=" + batchSize +
      '}';
  }
}
