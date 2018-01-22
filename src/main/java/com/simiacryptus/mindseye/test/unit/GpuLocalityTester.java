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
import com.simiacryptus.mindseye.lang.cudnn.CudaPtr;
import com.simiacryptus.mindseye.lang.cudnn.GpuHandle;
import com.simiacryptus.mindseye.lang.cudnn.GpuTensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.test.SimpleGpuEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Batching tester.
 */
public class GpuLocalityTester implements ComponentTest<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(GpuLocalityTester.class);
  
  private final double tolerance;
  private int batchSize = 10;
  
  /**
   * Instantiates a new Batching tester.
   *
   * @param tolerance the tolerance
   */
  public GpuLocalityTester(final double tolerance) {
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
    return GpuHandle.run(gpu -> {
      final TensorList[] heapInput = Arrays.stream(inputPrototype).map(t ->
                                                                         new TensorArray(IntStream.range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom()))
                                                                                                  .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
      TensorList[] gpuInput = Arrays.stream(heapInput).map(original -> {
        CudaPtr cudaPtr = CudaPtr.getCudaPtr(Precision.Double, original);
        return GpuTensorList.create(cudaPtr, original.length(), original.getDimensions(), Precision.Double);
      }).toArray(i -> new TensorList[i]);
      final SimpleResult fromHeap = SimpleGpuEval.run(reference, gpu, heapInput);
      final SimpleResult fromGPU = SimpleGpuEval.run(reference, gpu, gpuInput);
      
      final ToleranceStatistics outputAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch ->
                                                                                                new ToleranceStatistics().accumulate(
                                                                                                  fromHeap.getOutput().get(batch).getData(),
                                                                                                  fromGPU.getOutput().get(batch).getData())
                                                                                             ).reduce((a, b) -> a.combine(b)).get();
      if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
        logger.info("Batch Output: " + fromHeap.getOutput().stream().map(x -> x.prettyPrint()).collect(Collectors.toList()));
        logger.info("Singular Output: " + fromGPU.getOutput().stream().map(x -> x.prettyPrint()).collect(Collectors.toList()));
        throw new AssertionError("Output Corrupt: " + outputAgreement);
      }
      
      final ToleranceStatistics derivativeAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
        IntFunction<ToleranceStatistics> statisticsFunction = input ->
          new ToleranceStatistics().accumulate(
            fromHeap.getDerivative()[input].get(batch).getData(),
            fromGPU.getDerivative()[input].get(batch).getData());
        return IntStream.range(0, heapInput.length).mapToObj(statisticsFunction).reduce((a, b) -> a.combine(b)).get();
      }).reduce((a, b) -> a.combine(b)).get();
      if (!(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
        throw new AssertionError("Derivatives Corrupt: " + derivativeAgreement);
      }
      
      return derivativeAgreement.combine(outputAgreement);
    });
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
    log.h1("Multi-GPU Compatibility");
    log.p("This layer should be able to run using a GPU context other than the one used to create the inputs.");
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
  public GpuLocalityTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
}
