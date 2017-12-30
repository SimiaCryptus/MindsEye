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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.SimpleListEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Batching tester.
 */
public class BatchingTester implements ComponentTest<ToleranceStatistics> {
  
  private final double tolerance;
  
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
  public ToleranceStatistics test(final NNLayer reference, final Tensor[] inputPrototype) {
    if (null == reference) return new ToleranceStatistics();
  
    final int batchSize = 10;
    final TensorList[] inputTensorLists = Arrays.stream(inputPrototype).map(t ->
                                                                              new TensorArray(IntStream.range(0, batchSize).mapToObj(i -> t.map(v -> getRandom()))
                                                                                                       .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
    final SimpleListEval asABatch = SimpleListEval.run(reference, inputTensorLists);
    final List<SimpleEval> oneAtATime = IntStream.range(0, batchSize).mapToObj(batch ->
                                                                                 SimpleEval.run(reference, IntStream.range(0, inputTensorLists.length)
                                                                                                                    .mapToObj(i -> inputTensorLists[i].get(batch)).toArray(i -> new Tensor[i]))
                                                                              ).collect(Collectors.toList());
    
    final ToleranceStatistics outputAgreement = IntStream.range(0, batchSize).mapToObj(batch ->
                                                                                         new ToleranceStatistics().accumulate(
                                                                                           asABatch.getOutput().get(batch).getData(),
                                                                                           oneAtATime.get(batch).getOutput().getData())
                                                                                      ).reduce((a, b) -> a.combine(b)).get();
    if (!(outputAgreement.absoluteTol.getMax() < tolerance)) throw new AssertionError(outputAgreement.toString());
  
    final ToleranceStatistics derivativeAgreement = IntStream.range(0, batchSize).mapToObj(batch ->
                                                                                             IntStream.range(0, inputTensorLists.length).mapToObj(input ->
                                                                                                                                                    new ToleranceStatistics().accumulate(
                                                                                                                                                      asABatch.getDerivative()[input].get(batch).getData(),
                                                                                                                                                      oneAtATime.get(batch).getDerivative()[input].getData())
                                                                                                                                                 ).reduce((a, b) -> a.combine(b)).get()).reduce((a, b) -> a.combine(b)).get();
    if (!(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
      throw new AssertionError(derivativeAgreement.toString());
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
  public ToleranceStatistics test(final NotebookOutput log, final NNLayer reference, final Tensor... inputPrototype) {
    log.h1("Batch Execution");
    log.p("Most layers, including this one, should behave the same no matter how the items are split between batches. We verify this:");
    return log.code(() -> {
      return test(reference, inputPrototype);
    });
  }
  
}
