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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionMode;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This convolution layer only supports an equal number of input and output bands. It is used as the foundational
 * component for ConvolutionLayer, since the CudaSystem api has this restriction (in recent versions).
 */
@SuppressWarnings("serial")
public class SimpleConvolutionLayer extends LayerBase implements MultiPrecision<SimpleConvolutionLayer> {
  
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(SimpleConvolutionLayer.class);
  /**
   * The Kernel.
   */
  public final Tensor kernel;
  /**
   * The Filter.
   */
  @Nullable
  private final Map<Integer, CudaMemory> gpuFilters = new ConcurrentHashMap<>();
  private int paddingX;
  private int paddingY;
  private Precision precision = Precision.Double;
  private int strideX = 1;
  private int strideY = 1;
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected SimpleConvolutionLayer() {
    this(null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public SimpleConvolutionLayer(final int width, final int height, final int bands) {
    this(new Tensor(width, height, bands));
    kernel.freeRef();
    assert !false || 0 == (width - 1) % 2 : "Simple kernels must have odd width";
    assert !false || 0 == (height - 1) % 2 : "Simple kernels must have odd height";
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected SimpleConvolutionLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"), resources);
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    setPaddingX(json.get("paddingX").getAsInt());
    setPaddingY(json.get("paddingY").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the filter
   */
  protected SimpleConvolutionLayer(@javax.annotation.Nonnull final Tensor kernel) {
    super();
    @javax.annotation.Nonnull int[] kernelSize = kernel.getDimensions();
    if (kernelSize.length != 3) throw new IllegalArgumentException();
    if (kernelSize[0] <= 0) throw new IllegalArgumentException();
    if (kernelSize[1] <= 0) throw new IllegalArgumentException();
    if (kernelSize[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
    this.kernel.addRef(this);
    this.setPaddingX((int) Math.ceil((kernelSize[0] - 1) / 2.0));
    this.setPaddingY((int) Math.ceil((kernelSize[1] - 1) / 2.0));
    
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the convolution layer
   */
  public static SimpleConvolutionLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SimpleConvolutionLayer(json, rs);
  }
  
  /**
   * Reverse int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  @javax.annotation.Nonnull
  public static int[] reverse(@javax.annotation.Nonnull int... array) {
    for (int i = 0; i < array.length / 2; i++) {
      int j = array[array.length - (i + 1)];
      array[array.length - (i + 1)] = array[i];
      array[i] = j;
    }
    return array;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer addWeights(@javax.annotation.Nonnull final DoubleSupplier f) {
    Util.add(f, kernel.getData());
    return this;
  }
  
  private boolean cmp(final int[] outputSize, @javax.annotation.Nonnull final int[] outputDims) {
    if (4 != outputDims.length) return false;
    if (outputSize[0] != outputDims[3]) return false;
    if (outputSize[1] != outputDims[2]) return false;
    return outputSize[2] == outputDims[1];
  }
  
  @javax.annotation.Nullable
  @Override
  public Result evalAndFree(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
  
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @javax.annotation.Nonnull final int[] kernelSize = kernel.getDimensions();
    final int[] outputSize = getOutputSize(inputSize);
    final int length = inputData.length();
    kernel.addRef();
    SimpleConvolutionLayer.this.addRef();
    return new Result(CudaSystem.eval(gpu -> {
  
  
      @javax.annotation.Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device);
      final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(
        precision, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision,
        paddingY, paddingX,
        strideY, strideX,
        1, 1);
      final int[] outputDims = IntStream.of(reverse(CudaSystem.getOutputDims(inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr()))).limit(3).toArray();
      final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
        outputDims[2], outputDims[1], outputDims[0],
        outputDims[2] * outputDims[1] * outputDims[0], outputDims[1] * outputDims[0], outputDims[0], 1);
      final int forwardAlgorithm = gpu.getForwardAlgorithm(
        inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(),
        outputDescriptor.getPtr(), CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit());
      final CudaMemory forwardWorkspace = gpu.allocateForwardWorkspace(gpu,
        inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(),
        outputDescriptor.getPtr(), forwardAlgorithm);
      try {
        assert 0 < kernel.getData().length;
        assert kernelSize[0] * kernelSize[1] * kernelSize[2] == kernel.getData().length;
        @Nonnull CudaMemory filterPtr = getCudaFilter(gpu);
        @javax.annotation.Nonnull final CudaMemory outputBuffer = gpu.allocate(
          (long) Tensor.length(outputDims) * length * precision.size, MemoryType.Managed, true);
        CudaSystem.handle(gpu.cudnnConvolutionForward(precision.getPointer(1.0),
          inputTensor.descriptor.getPtr(), inputTensor.memory.getPtr(),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          convolutionDescriptor.getPtr(),
          forwardAlgorithm,
          null == forwardWorkspace ? null : forwardWorkspace.getPtr(),
          null == forwardWorkspace ? 0 : forwardWorkspace.size,
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputBuffer.getPtr()));
        filterPtr.freeRef();
        outputDescriptor.addRef();
        return CudaTensorList.wrap(CudaTensor.wrap(outputBuffer, outputDescriptor, precision), length, outputDims, precision);
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException(String.format("Error in convolution %s x %s", Arrays.toString(inputSize), Arrays.toString(kernelSize)), e);
      } finally {
        Stream.of(inputTensor, filterDescriptor, outputDescriptor, forwardWorkspace, convolutionDescriptor).forEach(ReferenceCounting::freeRef);
      }
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      delta.assertAlive();
      buffer.assertAlive();
      inputData.assertAlive();
      assert delta.length() == inputData.length();
      TestUtil.runAllSerial(() -> {
        CudaSystem.run(gpu -> {
          if (!isFrozen()) {
            @javax.annotation.Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device).getDenseAndFree(gpu);
            @javax.annotation.Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device).getDenseAndFree(gpu);
            final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(
              precision, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
            final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision,
              paddingY, paddingX,
              strideY, strideX,
              1, 1);
            final int backwardFilterAlgorithm = gpu.getBackwardFilterAlgorithm(
              inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(), CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit());
            final CudaMemory backwardsFilterWorkSpace = gpu.allocateBackwardFilterWorkspace(gpu,
              inputTensor.descriptor.getPtr(), filterDescriptor.getPtr(),
              convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(), backwardFilterAlgorithm);
            try {
              @javax.annotation.Nonnull CudaMemory filterPtr = gpu.allocate((long) kernel.length() * precision.size, MemoryType.Device, true);
              try {
                CudaSystem.handle(gpu.cudnnConvolutionBackwardFilter(precision.getPointer(1.0),
                  inputTensor.descriptor.getPtr(), inputTensor.memory.getPtr(),
                  deltaTensor.descriptor.getPtr(), deltaTensor.memory.getPtr(),
                  convolutionDescriptor.getPtr(),
                  backwardFilterAlgorithm,
                  backwardsFilterWorkSpace.getPtr(),
                  backwardsFilterWorkSpace.size,
                  precision.getPointer(0.0), filterDescriptor.getPtr(), filterPtr.getPtr()));
              } catch (@javax.annotation.Nonnull final Throwable e) {
                throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
              }
              @javax.annotation.Nonnull final Tensor weightGradient = filterPtr.read(precision, kernel.getDimensions());
              inputTensor.freeRef();
              filterPtr.freeRef();
              deltaTensor.freeRef();
              buffer.get(SimpleConvolutionLayer.this, kernel.getData()).addInPlace(weightGradient.getData()).freeRef();
              weightGradient.freeRef();
              clearCudaFilters();
            } finally {
              Stream.of(filterDescriptor, convolutionDescriptor, backwardsFilterWorkSpace).forEach(ReferenceCounting::freeRef);
            }
          }
        });
      }, () -> {
        if (input.isAlive()) {
          final TensorList inputBufferTensors = CudaSystem.eval(gpu -> {
            final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision, length, inputSize[2], inputSize[1], inputSize[0], inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1);
            final CudaResource<cudnnFilterDescriptor> filterDescriptor = gpu.newFilterDescriptor(
              precision, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
            final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = gpu.newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision,
              paddingY, paddingX,
              strideY, strideX,
              1, 1);
            @javax.annotation.Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device).getDenseAndFree(gpu);
            final int backwardDataAlgorithm = gpu.getBackwardDataAlgorithm(
              inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(), CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit());
            final CudaMemory backwardsDataWorkSpace = gpu.allocateBackwardDataWorkspace(gpu,
              inputDescriptor.getPtr(), filterDescriptor.getPtr(),
              convolutionDescriptor.getPtr(), deltaTensor.descriptor.getPtr(), backwardDataAlgorithm);
            try {
              @javax.annotation.Nonnull final CudaMemory inputBuffer = gpu.allocate((long) Tensor.length(inputData.getDimensions()) * length * precision.size, MemoryType.Device, true);
              try {
                @Nonnull final CudaMemory filterPtr = getCudaFilter(gpu);
                try {
                  CudaSystem.handle(gpu.cudnnConvolutionBackwardData(precision.getPointer(1.0),
                    filterDescriptor.getPtr(), filterPtr.getPtr(),
                    deltaTensor.descriptor.getPtr(), deltaTensor.memory.getPtr(),
                    convolutionDescriptor.getPtr(),
                    backwardDataAlgorithm,
                    backwardsDataWorkSpace.getPtr(),
                    backwardsDataWorkSpace.size,
                    precision.getPointer(0.0), inputDescriptor.getPtr(), inputBuffer.getPtr()));
                  inputDescriptor.addRef();
                  return CudaTensorList.wrap(CudaTensor.wrap(inputBuffer, inputDescriptor, precision), length, inputSize, precision);
                } finally {
                  filterPtr.freeRef();
                  deltaTensor.freeRef();
                }
              } catch (@javax.annotation.Nonnull final Throwable e) {
                throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
              }
            } finally {
              Stream.of(inputDescriptor, filterDescriptor, convolutionDescriptor, backwardsDataWorkSpace).forEach(ReferenceCounting::freeRef);
            }
          });
          if (null != inputBufferTensors) {
            input.accumulate(buffer, inputBufferTensors);
          }
        }
      });
    }) {
      
      @Override
      protected void _free() {
        kernel.freeRef();
        inputData.freeRef();
        Arrays.stream(inObj).forEach(ReferenceCounting::freeRef);
        SimpleConvolutionLayer.this.freeRef();
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  /**
   * Evict device data long.
   *
   * @param deviceId the device id
   * @return the long
   */
  public long evictDeviceData(final int deviceId) {
    CudaMemory remove = gpuFilters.remove(deviceId);
    if (null != remove) {
      if (1 == remove.currentRefCount()) {
        remove.freeRef();
        return remove.size;
      }
      else {
        CudaMemory race = gpuFilters.put(deviceId, remove);
        if (race != null) race.freeRef();
        return 0;
      }
    }
    else {
      return 0;
    }
  }
  
  
  @Nonnull
  private synchronized CudaMemory getCudaFilter(final CudnnHandle deviceNumber) {
    CudaMemory cudaMemory;
    if (!gpuFilters.containsKey(deviceNumber)) {
      double[] data = kernel.getData();
      cudaMemory = deviceNumber.allocate((long) data.length * precision.size, MemoryType.Device, true).write(precision, data);
      CudaMemory replaced = gpuFilters.put(deviceNumber.getDeviceId(), cudaMemory);
      if (null != replaced) replaced.freeRef();
    }
    else {
      cudaMemory = gpuFilters.get(deviceNumber);
    }
    cudaMemory.addRef();
    return cudaMemory;
  }
  
  @Nonnull
  private void clearCudaFilters() {
    gpuFilters.keySet().stream().collect(Collectors.toList()).stream().forEach(i -> {
      CudaMemory cudaMemory = gpuFilters.remove(i);
      if (null != cudaMemory) cudaMemory.freeRef();
    });
  }
  
  @Override
  protected void _free() {
    kernel.freeRef(this);
    clearCudaFilters();
    super._free();
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    log.info(String.format("Using compatibility layer for %s", this));
    int bands = (int) Math.sqrt(this.kernel.getDimensions()[2]);
    @javax.annotation.Nonnull final com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer convolutionLayer = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(this.kernel.getDimensions()[0], this.kernel.getDimensions()[1], this.kernel.getDimensions()[2], true);
    @javax.annotation.Nonnull final Tensor tensor = new Tensor(kernel.getDimensions());
    tensor.setByCoord(c -> {
      final int band = c.getCoords()[2];
      final int bandX = band % bands;
      final int bandY = (band - bandX) / bands;
      assert band == bandX + bandY * bands;
      final int bandT = bandY + bandX * bands;
      return kernel.get(c.getCoords()[0], c.getCoords()[1], bandT);
    });
    convolutionLayer.kernel.set(tensor);
    return new LayerBase() {
      @javax.annotation.Nonnull
      @Override
      public Result eval(@javax.annotation.Nonnull Result... array) {
        Arrays.stream(array).forEach(x -> x.addRef());
        @Nonnull Result result = convolutionLayer.eval(array);
        return new Result(result.getData(), (DeltaSet<Layer> buffer, TensorList data) -> {
          throw new IllegalStateException();
        }) {
  
  
          @Override
          protected void _free() {
            Arrays.stream(array).forEach(x -> x.freeRef());
          }
  
          @Override
          public boolean isAlive() {
            return false;
          }
        };
      }
  
      @javax.annotation.Nonnull
      @Override
      public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
        throw new IllegalStateException();
      }
  
      @javax.annotation.Nonnull
      @Override
      public List<double[]> state() {
        throw new IllegalStateException();
      }
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @javax.annotation.Nonnull DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    JsonElement value;
    try {
      value = kernel.toJson(resources, dataSerializer);
    } catch (Throwable e) {
      throw new RuntimeException("Error serializing convolution" + Arrays.toString(this.kernel.getDimensions()), e);
    }
    json.add("filter", value);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("precision", precision.name());
    return json;
  }
  
  /**
   * Get output size int [ ].
   *
   * @param inputSize the input size
   * @return the int [ ]
   */
  public int[] getOutputSize(final int... inputSize) {
    @javax.annotation.Nonnull final int[] kernelSize = kernel.getDimensions();
    try {
      return IntStream.range(0, kernelSize.length).map(i -> {
        int x;
        if (i == kernelSize.length - 1) {
          //assert kernelSize[i] == inputSize[i];
          x = kernelSize[i] / inputSize[i];
        }
        else {
          int padding;
          if (i == 0) {
            padding = this.paddingX;
          }
          else if (i == 1) {
            padding = this.paddingY;
          }
          else {
            throw new IllegalStateException();
          }
          x = inputSize[i] - (kernelSize[i] - 1) + padding * 2;
        }
        assert 0 < x;
        return x;
      }).toArray();
    } catch (Throwable e) {
      throw new RuntimeException(String.format("Error with convolution %s x %s (%s)", Arrays.toString(inputSize), Arrays.toString(kernelSize), getName()), e);
    }
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public SimpleConvolutionLayer setPrecision(final Precision precision) {
    clearCudaFilters();
    this.precision = precision;
    return this;
  }
  
  /**
   * The Stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }
  
  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   * @return the stride x
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this;
  }
  
  /**
   * The Stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }
  
  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   * @return the stride y
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer set(@javax.annotation.Nonnull final DoubleSupplier f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer set(@javax.annotation.Nonnull final ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).parallel().forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(kernel.getData());
  }
  
  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setPaddingX(int paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setPaddingY(int paddingY) {
    this.paddingY = paddingY;
    return this;
  }
  
  /**
   * Sets padding xy.
   *
   * @param x the x
   * @param y the y
   * @return the padding xy
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setPaddingXY(int x, int y) {
    return setPaddingX(x).setPaddingY(y);
  }
  
  /**
   * Sets weights log.
   *
   * @param f the f
   * @return the weights log
   */
  @javax.annotation.Nonnull
  public SimpleConvolutionLayer setWeightsLog(double f) {
    return set(() -> Math.pow(10, f) * (Math.random() - 0.5));
  }
  
  /**
   * Set.
   *
   * @param kernel the kernel
   */
  public void set(@javax.annotation.Nonnull Tensor kernel) {
    this.kernel.set(kernel);
  }
  
  /**
   * Get kernel dimensions int [ ].
   *
   * @return the int [ ]
   */
  public int[] getKernelDimensions() {
    return kernel.getDimensions();
  }
  
  
}
