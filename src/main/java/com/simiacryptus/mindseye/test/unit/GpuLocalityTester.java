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
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.test.SimpleGpuEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Batching tester.
 */
public class GpuLocalityTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(GpuLocalityTester.class);
  
  private final double tolerance;
  private int batchSize = 1;
  
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
  public ToleranceStatistics test(@Nullable final Layer reference, @javax.annotation.Nonnull final Tensor[] inputPrototype) {
    if (null == reference) return new ToleranceStatistics();
    return CudaSystem.eval(gpu -> {
      final TensorList[] heapInput = Arrays.stream(inputPrototype).map(t ->
        TensorArray.wrap(IntStream.range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom()))
          .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
      TensorList[] gpuInput = Arrays.stream(heapInput).map(original -> {
        @javax.annotation.Nullable CudaMemory cudaMemory = gpu.getPtr(original, Precision.Double, MemoryType.Managed);
        return CudaTensorList.wrap(cudaMemory, original.length(), original.getDimensions(), Precision.Double);
      }).toArray(i -> new TensorList[i]);
      @Nonnull final SimpleResult fromHeap = SimpleGpuEval.run(reference, gpu, heapInput);
      @Nonnull final SimpleResult fromGPU = SimpleGpuEval.run(reference, gpu, gpuInput);
      Arrays.stream(gpuInput).forEach(ReferenceCounting::freeRef);
  
      @javax.annotation.Nonnull final ToleranceStatistics outputAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
          Tensor a = fromHeap.getOutput().get(batch);
          Tensor b = fromGPU.getOutput().get(batch);
          ToleranceStatistics statistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
          a.freeRef();
          b.freeRef();
          return statistics;
        }
      ).reduce((a, b) -> a.combine(b)).get();
      if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
        logger.info("Batch Output: " + fromHeap.getOutput().stream().map(x -> {
          String str = x.prettyPrint();
          x.freeRef();
          return str;
        }).collect(Collectors.toList()));
        logger.info("Singular Output: " + fromGPU.getOutput().stream().map(x -> {
          String str = x.prettyPrint();
          x.freeRef();
          return str;
        }).collect(Collectors.toList()));
        throw new AssertionError("Output Corrupt: " + outputAgreement);
      }
  
      @javax.annotation.Nonnull final ToleranceStatistics derivativeAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
        @javax.annotation.Nonnull IntFunction<ToleranceStatistics> statisticsFunction = input -> {
          Tensor b = fromGPU.getDerivative()[input].get(batch);
          Tensor a = fromHeap.getDerivative()[input].get(batch);
          ToleranceStatistics statistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
          a.freeRef();
          b.freeRef();
          return statistics;
        };
        return IntStream.range(0, heapInput.length).mapToObj(statisticsFunction).reduce((a, b) -> a.combine(b)).get();
      }).reduce((a, b) -> a.combine(b)).get();
      fromGPU.freeRef();
      fromHeap.freeRef();
      Arrays.stream(heapInput).forEach(x -> x.freeRef());
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
  public ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput log, final Layer reference, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    log.h1("Multi-GPU Compatibility");
    log.p("This layer should be able to eval using a GPU context other than the one used to create the inputs.");
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
  @javax.annotation.Nonnull
  public GpuLocalityTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "GpuLocalityTester{" +
      "tolerance=" + tolerance +
      ", batchSize=" + batchSize +
      '}';
  }
}
