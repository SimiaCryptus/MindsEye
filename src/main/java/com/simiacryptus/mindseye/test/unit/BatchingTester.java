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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.SimpleListEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
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
  @Nonnull
  public ToleranceStatistics test(@Nullable final Layer reference, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference) return new ToleranceStatistics();

    final TensorList[] inputTensorLists = Arrays.stream(inputPrototype).map(t ->
        TensorArray.wrap(IntStream.range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom()))
            .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
    @Nonnull final SimpleResult asABatch;
    final List<SimpleEval> oneAtATime;
    try {
      asABatch = SimpleListEval.run(reference, inputTensorLists);
      oneAtATime = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
            Tensor[] inputTensors = IntStream.range(0, inputTensorLists.length)
                .mapToObj(i -> inputTensorLists[i].get(batch)).toArray(i -> new Tensor[i]);
            @Nonnull SimpleEval eval = SimpleEval.run(reference, inputTensors);
            for (@Nonnull Tensor tensor : inputTensors) {
              tensor.freeRef();
            }
            return eval;
          }
      ).collect(Collectors.toList());
    } finally {
      for (@Nonnull TensorList tensorList : inputTensorLists) {
        tensorList.freeRef();
      }
    }
    try {

      TensorList batchOutput = asABatch.getOutput();
      @Nonnull IntFunction<ToleranceStatistics> toleranceStatisticsIntFunction = batch -> {
        @Nullable Tensor batchTensor = batchOutput.get(batch);
        @Nonnull ToleranceStatistics accumulate = new ToleranceStatistics().accumulate(
            batchTensor.getData(),
            oneAtATime.get(batch).getOutput().getData());
        batchTensor.freeRef();
        return accumulate;
      };
      int batchLength = batchOutput.length();
      @Nonnull final ToleranceStatistics outputAgreement = IntStream.range(0, Math.min(getBatchSize(), batchLength))
          .mapToObj(toleranceStatisticsIntFunction)
          .reduce((a, b) -> a.combine(b)).get();
      if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
        logger.info("Batch Output: " + batchOutput.stream().map(x -> {
          String str = x.prettyPrint();
          x.freeRef();
          return str;
        }).collect(Collectors.toList()));
        logger.info("Singular Output: " + oneAtATime.stream().map(x -> x.getOutput().prettyPrint()).collect(Collectors.toList()));
        throw new AssertionError("Output Corrupt: " + outputAgreement);
      }

      ToleranceStatistics derivativeAgreement = IntStream.range(0, Math.min(getBatchSize(), batchLength)).mapToObj(batch -> {
        IntFunction<ToleranceStatistics> statisticsFunction = input -> {
          @Nullable Tensor a = asABatch.getInputDerivative()[input].get(batch);
          Tensor b = oneAtATime.get(batch).getDerivative()[input];
          @Nonnull Tensor diff = a.minus(b);
          logger.info("Error: " + diff.prettyPrint());
          logger.info("Scalar Statistics: " + new ScalarStatistics().add(diff.getData()).getMetrics());
          double[][] points = Arrays.stream(diff.getData()).mapToObj(x -> new double[]{x}).toArray(i -> new double[i][]);
          //logger.info("Density: " + new DensityTree("x").setMinSplitFract(1e-8).setSplitSizeThreshold(2).new Node(points));
          diff.freeRef();
          @Nonnull ToleranceStatistics toleranceStatistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
          a.freeRef();
          return toleranceStatistics;
        };
        return IntStream.range(0, Math.min(inputPrototype.length, batchLength)).mapToObj(statisticsFunction).reduce((a, b) -> a.combine(b)).orElse(null);
      }).filter(x -> x != null).reduce((a, b) -> a.combine(b)).orElse(null);

      if (null != derivativeAgreement && !(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
        throw new AssertionError("Derivatives Corrupt: " + derivativeAgreement);
      }
      return null != derivativeAgreement ? derivativeAgreement.combine(outputAgreement) : outputAgreement;
    } finally {
      asABatch.freeRef();
      oneAtATime.forEach(x -> x.freeRef());
    }
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
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, final Layer reference, @Nonnull final Tensor... inputPrototype) {
    log.h1("Batch Execution");
    log.p("Most layers, including this one, should behave the same no matter how the items are split between batches. We verify this:");
    return log.eval(() -> {
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
  @Nonnull
  public BatchingTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  @Nonnull
  @Override
  public String toString() {
    return "BatchingTester{" +
        "tolerance=" + tolerance +
        ", batchSize=" + batchSize +
        '}';
  }
}
