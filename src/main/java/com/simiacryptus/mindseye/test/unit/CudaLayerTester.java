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
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudnnHandle;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.test.SimpleGpuEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Batching tester.
 */
public class CudaLayerTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(CudaLayerTester.class);
  
  private final double tolerance;
  private int batchSize = 1;
  
  /**
   * Instantiates a new Batching tester.
   *
   * @param tolerance the tolerance
   */
  public CudaLayerTester(final double tolerance) {
    this.tolerance = tolerance;
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param log
   * @param layer          the reference
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput log, final Layer layer, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    log.h1("GPU/Cuda Behavior");
    layer.setFrozen(false);
    if (null == layer) return new ToleranceStatistics();
    ToleranceStatistics statistics = testInterGpu(log, layer, inputPrototype);
    try {
      statistics = statistics.combine(testNonstandardBounds(log, layer, inputPrototype));
      statistics = statistics.combine(testNonstandardBoundsBackprop(log, layer, inputPrototype));
    } catch (Throwable e) {
      logger.warn("Error testing support for tensor views", e);
      throw e;
    }
    return statistics;
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
   * Test inter gpu tolerance statistics.
   *
   * @param log            the log
   * @param reference      the reference
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics testInterGpu(final NotebookOutput log, @Nullable final Layer reference, @Nonnull final Tensor[] inputPrototype) {
    log.h2("Multi-GPU Compatibility");
    log.p("This layer should be able to eval using a GPU context other than the one used to create the inputs.");
    return log.code(() -> {
      final TensorList[] heapInput = Arrays.stream(inputPrototype).map(t ->
        TensorArray.wrap(IntStream.range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom()))
          .toArray(i -> new Tensor[i]))).toArray(i -> new TensorList[i]);
      logger.info("Input: " + Arrays.stream(heapInput).flatMap(x -> x.stream()).map(tensor -> {
        String prettyPrint = tensor.prettyPrint();
        tensor.freeRef();
        return prettyPrint;
      }).collect(Collectors.toList()));
      TensorList[] gpuInput = CudaSystem.eval(gpu -> {
        return Arrays.stream(heapInput).map(original -> {
          return CudaTensorList.wrap(gpu.getTensor(original, Precision.Double, MemoryType.Managed, false), original.length(), original.getDimensions(), Precision.Double);
        }).toArray(i -> new TensorList[i]);
      }, 0);
      @Nonnull final SimpleResult fromHeap = CudaSystem.eval(gpu -> SimpleGpuEval.run(reference, gpu, heapInput), 1);
      @Nonnull final SimpleResult fromGPU = CudaSystem.eval(gpu -> SimpleGpuEval.run(reference, gpu, gpuInput), 1);
      try {
        ToleranceStatistics compareOutput = compareOutput(fromHeap, fromGPU);
        ToleranceStatistics compareDerivatives = compareDerivatives(fromHeap, fromGPU);
        return compareDerivatives.combine(compareOutput);
      } finally {
        Arrays.stream(gpuInput).forEach(ReferenceCounting::freeRef);
        Arrays.stream(heapInput).forEach(x -> x.freeRef());
        fromGPU.freeRef();
        fromHeap.freeRef();
      }
    });
  }
  
  /**
   * Test nonstandard bounds tolerance statistics.
   *
   * @param log            the log
   * @param reference      the reference
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics testNonstandardBounds(final NotebookOutput log, @Nullable final Layer reference, @Nonnull final Tensor[] inputPrototype) {
    log.h2("Irregular Input");
    log.p("This layer should be able to accept non-dense inputs.");
    return log.code(() -> {
      Tensor[] randomized = Arrays.stream(inputPrototype).map(x -> x.map(v -> getRandom())).toArray(i -> new Tensor[i]);
      logger.info("Input: " + Arrays.stream(randomized).map(Tensor::prettyPrint).collect(Collectors.toList()));
      Precision precision = Precision.Double;
      
      TensorList[] controlInput = CudaSystem.eval(gpu -> {
        return Arrays.stream(randomized).map(original -> {
          TensorArray data = TensorArray.create(original);
          CudaTensorList wrap = CudaTensorList.wrap(gpu.getTensor(data, precision, MemoryType.Managed, false), 1, original.getDimensions(), precision);
          data.freeRef();
          return wrap;
        }).toArray(i -> new TensorList[i]);
      }, 0);
      @Nonnull final SimpleResult controlResult = CudaSystem.eval(gpu -> SimpleGpuEval.run(reference, gpu, controlInput), 1);
      
      final TensorList[] irregularInput = CudaSystem.eval(gpu -> {
        return Arrays.stream(randomized).map(original -> {
          return buildIrregularCudaTensor(gpu, precision, original);
        }).toArray(i -> new TensorList[i]);
      }, 0);
      @Nonnull final SimpleResult testResult = CudaSystem.eval(gpu -> SimpleGpuEval.run(reference, gpu, irregularInput), 1);
      
      try {
        ToleranceStatistics compareOutput = compareOutput(controlResult, testResult);
        ToleranceStatistics compareDerivatives = compareDerivatives(controlResult, testResult);
        return compareDerivatives.combine(compareOutput);
      } finally {
        Arrays.stream(randomized).forEach(ReferenceCountingBase::freeRef);
        Arrays.stream(controlInput).forEach(ReferenceCounting::freeRef);
        Arrays.stream(irregularInput).forEach(x -> x.freeRef());
        controlResult.freeRef();
        testResult.freeRef();
      }
    });
  }
  
  /**
   * Test nonstandard bounds backprop tolerance statistics.
   *
   * @param log            the log
   * @param layer          the layer
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics testNonstandardBoundsBackprop(final NotebookOutput log, @Nullable final Layer layer, @Nonnull final Tensor[] inputPrototype) {
    log.h2("Irregular Backprop");
    log.p("This layer should accept non-dense tensors as delta input.");
    return log.code(() -> {
      Tensor[] randomized = Arrays.stream(inputPrototype).map(x -> x.map(v -> getRandom())).toArray(i -> new Tensor[i]);
      logger.info("Input: " + Arrays.stream(randomized).map(Tensor::prettyPrint).collect(Collectors.toList()));
      Precision precision = Precision.Double;
      TensorList[] controlInput = Arrays.stream(randomized).map(original -> {
        return TensorArray.wrap(original);
      }).toArray(i -> new TensorList[i]);
      @Nonnull final SimpleResult testResult = CudaSystem.eval(gpu -> {
        TensorList[] copy = copy(controlInput);
        SimpleResult result = new SimpleGpuEval(layer, gpu, copy) {
          @Nonnull
          @Override
          public TensorList getFeedback(@Nonnull final TensorList original) {
            Tensor originalTensor = original.get(0).mapAndFree(x -> 1);
            CudaTensorList cudaTensorList = buildIrregularCudaTensor(gpu, precision, originalTensor);
            originalTensor.freeRef();
            return cudaTensorList;
          }
        }.call();
        Arrays.stream(copy).forEach(ReferenceCounting::freeRef);
        return result;
      });
      @Nonnull final SimpleResult controlResult = CudaSystem.eval(gpu -> {
        TensorList[] copy = copy(controlInput);
        SimpleResult result = SimpleGpuEval.run(layer, gpu, copy);
        Arrays.stream(copy).forEach(ReferenceCounting::freeRef);
        return result;
      }, 1);
      try {
        ToleranceStatistics compareOutput = compareOutput(controlResult, testResult);
        ToleranceStatistics compareDerivatives = compareDerivatives(controlResult, testResult);
        return compareDerivatives.combine(compareOutput);
      } finally {
        Arrays.stream(controlInput).forEach(ReferenceCounting::freeRef);
        controlResult.freeRef();
        testResult.freeRef();
      }
    });
  }
  
  /**
   * Copy tensor list [ ].
   *
   * @param controlInput the control input
   * @return the tensor list [ ]
   */
  public TensorList[] copy(final TensorList[] controlInput) {
    return Arrays.stream(controlInput).map(x -> x.copy()).toArray(i -> new TensorList[i]);
  }
  
  /**
   * Build irregular cuda tensor cuda tensor list.
   *
   * @param gpu       the gpu
   * @param precision the precision
   * @param original  the original
   * @return the cuda tensor list
   */
  public CudaTensorList buildIrregularCudaTensor(final CudnnHandle gpu, final Precision precision, final Tensor original) {
    TensorArray data = TensorArray.create(original);
    int[] inputSize = original.getDimensions();
    int channels = inputSize.length < 3 ? 1 : inputSize[2];
    int height = inputSize.length < 2 ? 1 : inputSize[1];
    int width = inputSize.length < 1 ? 1 : inputSize[0];
    final int listLength = 1;
    final int elementLength = data.getElements();
  
    MemoryType memoryType = MemoryType.Managed;
    @Nonnull final CudaMemory ptr0 = gpu.allocate((long) elementLength * listLength * precision.size, memoryType, false);
    @Nonnull final CudaDevice.CudaTensorDescriptor descriptor0 = gpu.newTensorDescriptor(precision,
      listLength, channels, height, width,
      channels * height * width, height * width, width, 1);
    for (int i = 0; i < listLength; i++) {
      Tensor tensor = data.get(i);
      assert null != data;
      assert null != tensor;
      assert Arrays.equals(tensor.getDimensions(), data.getDimensions()) : Arrays.toString(tensor.getDimensions()) + " != " + Arrays.toString(data.getDimensions());
      ptr0.write(precision, tensor.getData(), (long) i * elementLength);
      tensor.freeRef();
    }
    data.freeRef();
    Random r = new Random();
    int c = r.nextInt(5);
    int v = r.nextInt(5);
    int h = r.nextInt(5);
    @Nonnull final CudaMemory ptr1 = gpu.allocate((long) (channels + c) * (height + v) * (width + h) * listLength * precision.size, memoryType, false);
    @Nonnull final CudaDevice.CudaTensorDescriptor descriptor1 = gpu.newTensorDescriptor(precision,
      listLength, channels, height, width,
      (height + v) * (width + h) * (channels + c), (height + v) * (width + h), width + h, 1);
    gpu.cudnnTransformTensor(
      precision.getPointer(1.0), descriptor0.getPtr(), ptr0.getPtr(),
      precision.getPointer(0.0), descriptor1.getPtr(), ptr1.getPtr()
    );
    descriptor0.freeRef();
    ptr0.freeRef();
    return CudaTensorList.wrap(CudaTensor.wrap(ptr1, descriptor1, precision), 1, original.getDimensions(), precision);
  }
  
  /**
   * Compare derivatives tolerance statistics.
   *
   * @param expected the expected
   * @param actual   the actual
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics compareDerivatives(final SimpleResult expected, final SimpleResult actual) {
    ToleranceStatistics derivativeAgreement = compareInputDerivatives(expected, actual);
    derivativeAgreement = derivativeAgreement.combine(compareLayerDerivatives(expected, actual));
    return derivativeAgreement;
  }
  
  /**
   * Compare layer derivatives tolerance statistics.
   *
   * @param expected the expected
   * @param actual   the actual
   * @return the tolerance statistics
   */
  @Nullable
  public ToleranceStatistics compareLayerDerivatives(final SimpleResult expected, final SimpleResult actual) {
    @Nonnull final ToleranceStatistics derivativeAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
      @Nonnull Function<Layer, ToleranceStatistics> compareInputDerivative = input -> {
        double[] b = actual.getLayerDerivative().getMap().get(input).getDelta();
        double[] a = expected.getLayerDerivative().getMap().get(input).getDelta();
        ToleranceStatistics statistics = new ToleranceStatistics().accumulate(a, b);
        return statistics;
      };
      return Stream.concat(
        actual.getLayerDerivative().getMap().keySet().stream(),
        expected.getLayerDerivative().getMap().keySet().stream()
      ).distinct().map(compareInputDerivative).reduce((a, b) -> a.combine(b));
    }).filter(x -> x.isPresent()).map(x -> x.get()).reduce((a, b) -> a.combine(b)).orElse(null);
    if (null != derivativeAgreement && !(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
      logger.info("Expected Derivative: " + Arrays.stream(expected.getInputDerivative()).flatMap(TensorList::stream).map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      logger.info("Actual Derivative: " + Arrays.stream(actual.getInputDerivative()).flatMap(TensorList::stream).map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      throw new AssertionError("Layer Derivatives Corrupt: " + derivativeAgreement);
    }
    return derivativeAgreement;
  }
  
  /**
   * Compare input derivatives tolerance statistics.
   *
   * @param expected the expected
   * @param actual   the actual
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics compareInputDerivatives(final SimpleResult expected, final SimpleResult actual) {
    @Nonnull final ToleranceStatistics derivativeAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
      @Nonnull IntFunction<ToleranceStatistics> compareInputDerivative = input -> {
        Tensor b = actual.getInputDerivative()[input].get(batch);
        Tensor a = expected.getInputDerivative()[input].get(batch);
        ToleranceStatistics statistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
        a.freeRef();
        b.freeRef();
        return statistics;
      };
      return IntStream.range(0, expected.getOutput().length()).mapToObj(compareInputDerivative).reduce((a, b) -> a.combine(b)).get();
    }).reduce((a, b) -> a.combine(b)).get();
    if (!(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
      logger.info("Expected Derivative: " + Arrays.stream(expected.getInputDerivative()).flatMap(TensorList::stream).map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      logger.info("Actual Derivative: " + Arrays.stream(actual.getInputDerivative()).flatMap(TensorList::stream).map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      throw new AssertionError("Input Derivatives Corrupt: " + derivativeAgreement);
    }
    return derivativeAgreement;
  }
  
  /**
   * Compare output tolerance statistics.
   *
   * @param expected the expected
   * @param actual   the actual
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics compareOutput(final SimpleResult expected, final SimpleResult actual) {
    return compareOutput(expected.getOutput(), actual.getOutput());
  }
  
  /**
   * Compare output tolerance statistics.
   *
   * @param expectedOutput the expected output
   * @param actualOutput   the actual output
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics compareOutput(final TensorList expectedOutput, final TensorList actualOutput) {
    @Nonnull final ToleranceStatistics outputAgreement = IntStream.range(0, getBatchSize()).mapToObj(batch -> {
        Tensor a = expectedOutput.get(batch);
        Tensor b = actualOutput.get(batch);
        ToleranceStatistics statistics = new ToleranceStatistics().accumulate(a.getData(), b.getData());
        a.freeRef();
        b.freeRef();
        return statistics;
      }
    ).reduce((a, b) -> a.combine(b)).get();
    if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
      logger.info("Expected Output: " + expectedOutput.stream().map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      logger.info("Actual Output: " + actualOutput.stream().map(x -> {
        String str = x.prettyPrint();
        x.freeRef();
        return str;
      }).collect(Collectors.toList()));
      throw new AssertionError("Output Corrupt: " + outputAgreement);
    }
    return outputAgreement;
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
  public CudaLayerTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "CudaLayerTester{" +
      "tolerance=" + tolerance +
      ", batchSize=" + batchSize +
      '}';
  }
}
