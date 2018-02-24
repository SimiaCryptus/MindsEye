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
import jcuda.jcudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.DoubleSupplier;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
  private static final HashMap<SimpleConvolutionParameters, Supplier<CudaFwdParameters>> fwdWs = new HashMap<>();
  private static final HashMap<SimpleConvolutionParameters, Supplier<CudaRevParameters>> revWs = new HashMap<>();
  /**
   * The Filter.
   */
  @Nullable
  private final Map<Integer, CudaMemory> gpuFilters = new ConcurrentHashMap<>();
  /**
   * The Kernel.
   */
  public final Tensor kernel;
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
  
  @Nonnull
  private static CudaFwdParameters obtainFwd(@javax.annotation.Nonnull SimpleConvolutionParameters parameters) {
    Supplier<CudaFwdParameters> supplier = fwdWs.get(parameters);
    @Nullable CudaFwdParameters fwdParameters = supplier == null ? null : supplier.get();
    PersistanceMode workspaceCachePersistance = CudaSettings.INSTANCE.getWorkspaceCachePersistance();
    boolean isCached = workspaceCachePersistance != PersistanceMode.NULL;
    if (null == fwdParameters) {
      synchronized (fwdWs) {
        supplier = fwdWs.get(parameters);
        fwdParameters = supplier == null ? null : supplier.get();
        if (null == fwdParameters) {
          fwdParameters = new CudaFwdParameters(parameters);
          if (isCached) {
            fwdWs.put(parameters, workspaceCachePersistance.wrap(fwdParameters));
            fwdParameters.detach();
          }
        }
        if (isCached) fwdParameters.addRef();
        parameters.freeRef();
        return fwdParameters;
      }
    }
    if (isCached) fwdParameters.addRef();
    parameters.freeRef();
    return fwdParameters;
  }
  
  @Nonnull
  private static CudaRevParameters obtainRev(@javax.annotation.Nonnull SimpleConvolutionParameters parameters) {
    Supplier<CudaRevParameters> supplier = revWs.get(parameters);
    @Nullable CudaRevParameters revParameters = supplier == null ? null : supplier.get();
    PersistanceMode workspaceCachePersistance = CudaSettings.INSTANCE.getWorkspaceCachePersistance();
    boolean isCached = workspaceCachePersistance != PersistanceMode.NULL;
    if (null == revParameters) {
      synchronized (revWs) {
        supplier = revWs.get(parameters);
        revParameters = supplier == null ? null : supplier.get();
        if (null == revParameters) {
          revParameters = new CudaRevParameters(parameters);
          if (isCached) {
            revWs.put(parameters, workspaceCachePersistance.wrap(revParameters));
            revParameters.detach();
          }
        }
        if (isCached) revParameters.addRef();
        parameters.freeRef();
        return revParameters;
      }
    }
    if (isCached) revParameters.addRef();
    parameters.freeRef();
    return revParameters;
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
      @Nullable CudaFwdParameters cudaParameters = null;
      try {
        cudaParameters = obtainFwd(new SimpleConvolutionParameters(kernel, paddingX, paddingY, precision, strideX, strideY, length, inputSize, outputSize, kernelSize, gpu, CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit()));
        assert 0 < kernel.getData().length;
        assert kernelSize[0] * kernelSize[1] * kernelSize[2] == kernel.getData().length;
        @Nonnull CudaMemory filterPtr = getCudaFilter(gpu);
        @javax.annotation.Nullable final CudaMemory inputBuffer = gpu.getPtr(inputData, precision, MemoryType.Device);
        @javax.annotation.Nonnull final CudaMemory outputBuffer = gpu.allocate(Tensor.dim(cudaParameters.outputDims) * 1l * length * precision.size, MemoryType.Managed, true);
        CudaSystem.handle(gpu.cudnnConvolutionForward(precision.getPointer(1.0),
          cudaParameters.inputDescriptor.getPtr(), inputBuffer.getPtr(),
          cudaParameters.filterDescriptor.getPtr(), filterPtr.getPtr(),
          cudaParameters.convolutionDescriptor.getPtr(),
          cudaParameters.forwardAlgorithm,
          null == cudaParameters.forwardWorkspace ? null : cudaParameters.forwardWorkspace.getPtr(),
          null == cudaParameters.forwardWorkspace ? 0 : cudaParameters.forwardWorkspace.size,
          precision.getPointer(0.0), cudaParameters.outputDescriptor.getPtr(), outputBuffer.getPtr()));
        inputBuffer.freeRef();
        filterPtr.freeRef();
        return CudaTensorList.wrap(outputBuffer, length, cudaParameters.outputDims, precision);
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException(String.format("Error in convolution %s x %s", Arrays.toString(inputSize), Arrays.toString(kernelSize)), e);
      } finally {
        if (null != cudaParameters) cudaParameters.freeRef();
      }
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      delta.assertAlive();
      buffer.assertAlive();
      inputData.assertAlive();
      assert delta.length() == inputData.length();
      TestUtil.runAllSerial(() -> {
        CudaSystem.run(gpu -> {
          if (!isFrozen()) {
            @Nullable CudaRevParameters cudaParameters = obtainRev(new SimpleConvolutionParameters(kernel, paddingX, paddingY, precision, strideX, strideY, length, inputSize, outputSize, kernelSize, gpu, CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit()));
            assert cudaParameters.precision == precision;
            @javax.annotation.Nullable final CudaMemory errorPtr = gpu.getPtr(delta, precision, MemoryType.Device);
            @javax.annotation.Nullable final CudaMemory inputPtr = gpu.getPtr(inputData, precision, MemoryType.Device);
            @javax.annotation.Nonnull CudaMemory filterPtr = gpu.allocate(kernel.dim() * 1l * precision.size, MemoryType.Device, true);
            try {
              CudaSystem.handle(gpu.cudnnConvolutionBackwardFilter(cudaParameters.precision.getPointer(1.0),
                cudaParameters.inputDescriptor.getPtr(), inputPtr.getPtr(),
                cudaParameters.outputDescriptor.getPtr(), errorPtr.getPtr(),
                cudaParameters.convolutionDescriptor.getPtr(),
                cudaParameters.backwardFilterAlgorithm,
                cudaParameters.backwardsFilterWorkSpace.getPtr(),
                cudaParameters.backwardsFilterWorkSpace.size,
                precision.getPointer(0.0), cudaParameters.filterDescriptor.getPtr(), filterPtr.getPtr()));
            } catch (@javax.annotation.Nonnull final Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            }
            @javax.annotation.Nonnull final Tensor weightGradient = filterPtr.read(precision, kernel.getDimensions());
            inputPtr.freeRef();
            filterPtr.freeRef();
            cudaParameters.freeRef();
            errorPtr.freeRef();
            buffer.get(SimpleConvolutionLayer.this, kernel.getData()).addInPlace(weightGradient.getData()).freeRef();
            weightGradient.freeRef();
            clearCudaFilters();
          }
        });
      }, () -> {
        if (input.isAlive()) {
          final TensorList inputBufferTensors = CudaSystem.eval(gpu -> {
            @Nonnull CudaRevParameters cudaParameters = obtainRev(new SimpleConvolutionParameters(kernel, paddingX, paddingY, precision, strideX, strideY, length, inputSize, outputSize, kernelSize, gpu, CudaSettings.INSTANCE.getConvolutionWorkspaceSizeLimit()));
            assert cudaParameters != null;
            @javax.annotation.Nonnull final CudaMemory inputBuffer = gpu.allocate(Tensor.dim(inputData.getDimensions()) * 1l * length * precision.size, MemoryType.Device, true);
            try {
              @javax.annotation.Nullable final CudaMemory errorPtr = gpu.getPtr(delta, precision, MemoryType.Device);
              @Nonnull final CudaMemory filterPtr = getCudaFilter(gpu);
              try {
                CudaSystem.handle(gpu.cudnnConvolutionBackwardData(precision.getPointer(1.0),
                  cudaParameters.filterDescriptor.getPtr(), filterPtr.getPtr(),
                  cudaParameters.outputDescriptor.getPtr(), errorPtr.getPtr(),
                  cudaParameters.convolutionDescriptor.getPtr(),
                  cudaParameters.backwardDataAlgorithm,
                  cudaParameters.backwardsDataWorkSpace.getPtr(),
                  cudaParameters.backwardsDataWorkSpace.size,
                  precision.getPointer(0.0), cudaParameters.inputDescriptor.getPtr(), inputBuffer.getPtr()));
              } finally {
                filterPtr.freeRef();
                errorPtr.freeRef();
              }
            } catch (@javax.annotation.Nonnull final Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            } finally {
              cudaParameters.freeRef();
            }
            return CudaTensorList.wrap(inputBuffer, length, inputSize, precision);
          });
          try {
            if (null != inputBufferTensors) {
              input.accumulate(buffer, inputBufferTensors);
            }
          } finally {
            if (null != inputBufferTensors) {
              inputBufferTensors.freeRef();
            }
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
      throw new RuntimeException(String.format("Error with convolution %s x %s", Arrays.toString(inputSize), Arrays.toString(kernelSize)), e);
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
  
  private static class SimpleConvolutionParameters extends ReferenceCountingBase {
    /**
     * The Length.
     */
    public final int length;
    /**
     * The Input size.
     */
    @javax.annotation.Nonnull
    public final int[] inputSize;
    /**
     * The Output size.
     */
    @javax.annotation.Nonnull
    public final int[] outputSize;
    /**
     * The Kernel size.
     */
    @javax.annotation.Nonnull
    public final int[] kernelSize;
    /**
     * The Kernel.
     */
    public final Tensor kernel;
    /**
     * The Padding x.
     */
    public final int paddingX;
    /**
     * The Padding y.
     */
    public final int paddingY;
    /**
     * The Precision.
     */
    public final Precision precision;
    /**
     * The Stride x.
     */
    public final int strideX;
    /**
     * The Stride y.
     */
    public final int strideY;
    /**
     * The Gpu.
     */
    public final CudnnHandle gpu;
    /**
     * The Memory limit in bytes.
     */
    public int memoryLimitInBytes;
  
    /**
     * Instantiates a new Simple convolution parameters.
     *
     * @param kernel             the kernel
     * @param paddingX           the padding x
     * @param paddingY           the padding y
     * @param precision          the precision
     * @param strideX            the stride x
     * @param strideY            the stride y
     * @param length             the length
     * @param inputSize          the input size
     * @param outputSize         the output size
     * @param kernelSize         the kernel size
     * @param gpu                the gpu
     * @param memoryLimitInBytes the memory limit in bytes
     */
    public SimpleConvolutionParameters(Tensor kernel, int paddingX, int paddingY, Precision precision, int strideX, int strideY, int length, @javax.annotation.Nonnull int[] inputSize, @javax.annotation.Nonnull int[] outputSize, @javax.annotation.Nonnull int[] kernelSize, CudnnHandle gpu, final int memoryLimitInBytes) {
      this.paddingX = paddingX;
      this.gpu = gpu;
      this.strideX = strideX;
      this.strideY = strideY;
      this.paddingY = paddingY;
      this.precision = precision;
      this.kernel = kernel;
      this.kernel.addRef();
      this.kernel.detach();
      this.length = length;
      this.inputSize = Arrays.copyOf(inputSize, inputSize.length);
      this.outputSize = Arrays.copyOf(outputSize, outputSize.length);
      this.kernelSize = Arrays.copyOf(kernelSize, kernelSize.length);
      this.memoryLimitInBytes = memoryLimitInBytes;
    }
    
    @Override
    public void _free() {
      super._free();
      this.kernel.freeRef();
    }
  
    @Override
    public void detach() {
      this.kernel.detach();
      super.detach();
    }
  
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof SimpleConvolutionParameters)) return false;
      @javax.annotation.Nonnull SimpleConvolutionParameters that = (SimpleConvolutionParameters) o;
      if (length != that.length) return false;
      if (paddingX != that.paddingX) return false;
      if (paddingY != that.paddingY) return false;
      if (strideX != that.strideX) return false;
      if (strideY != that.strideY) return false;
      if (!gpu.equals(that.gpu)) return false;
      if (!Arrays.equals(inputSize, that.inputSize)) return false;
      if (!Arrays.equals(outputSize, that.outputSize)) return false;
      if (!Arrays.equals(kernelSize, that.kernelSize)) return false;
      if (kernel != null ? !kernel.equals(that.kernel) : that.kernel != null) return false;
      return precision == that.precision;
    }
    
    @Override
    public int hashCode() {
      int result = length;
      result = 31 * result + Arrays.hashCode(inputSize);
      result = 31 * result + Arrays.hashCode(outputSize);
      result = 31 * result + Arrays.hashCode(kernelSize);
      result = 31 * result + (kernel != null ? kernel.hashCode() : 0);
      result = 31 * result + paddingX;
      result = 31 * result + paddingY;
      result = 31 * result + (precision != null ? precision.hashCode() : 0);
      result = 31 * result + strideX;
      result = 31 * result + strideY;
      result = 31 * result + gpu.hashCode();
      return result;
    }
    
    @javax.annotation.Nonnull
    @Override
    public String toString() {
      return getClass().getSimpleName() + "{" +
        "length=" + length +
        ", inputSize=" + Arrays.toString(inputSize) +
        ", outputSize=" + Arrays.toString(outputSize) +
        ", kernelSize=" + Arrays.toString(kernelSize) +
        ", kernel=" + kernel +
        ", paddingX=" + paddingX +
        ", paddingY=" + paddingY +
        ", precision=" + precision +
        ", strideX=" + strideX +
        ", strideY=" + strideY +
        ", gpu=" + gpu +
        '}';
    }
  }
  
  private static class CudaRevParameters extends SimpleConvolutionParameters {
    /**
     * The Backward data algorithm.
     */
    public final int backwardDataAlgorithm;
    /**
     * The Backward filter algorithm.
     */
    public final int backwardFilterAlgorithm;
    /**
     * The Output dims.
     */
    public final int[] outputDims;
    /**
     * The Backwards work space.
     */
    @javax.annotation.Nonnull
    public final CudaMemory backwardsFilterWorkSpace;
    /**
     * The Output descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnTensorDescriptor> outputDescriptor;
    /**
     * The Input descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnTensorDescriptor> inputDescriptor;
    /**
     * The Filter descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnFilterDescriptor> filterDescriptor;
    /**
     * The Convolution descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor;
    /**
     * Backwards data workspace
     */
    @javax.annotation.Nonnull
    public final CudaMemory backwardsDataWorkSpace;
  
    /**
     * The Key.
     */
    public final SimpleConvolutionParameters key;
  
    /**
     * Instantiates a new Cuda rev parameters.
     *
     * @param obj the obj
     */
    CudaRevParameters(@javax.annotation.Nonnull SimpleConvolutionParameters obj) {
      super(obj.kernel, obj.paddingX, obj.paddingY, obj.precision, obj.strideX, obj.strideY, obj.length, obj.inputSize, obj.outputSize, obj.kernelSize, obj.gpu, obj.memoryLimitInBytes);
      key = obj;
      key.addRef();
      inputDescriptor = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      filterDescriptor = gpu.newFilterDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      convolutionDescriptor = gpu.newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision.code,
        paddingY, paddingX,
        strideY, strideX,
        1, 1);
      outputDims = IntStream.of(reverse(CudaSystem.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr()))).limit(3).toArray();
      outputDescriptor = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, outputDims[2], outputDims[1], outputDims[0]);
      backwardDataAlgorithm = gpu.getBackwardDataAlgorithm(
        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), obj.memoryLimitInBytes);
      backwardFilterAlgorithm = gpu.getBackwardFilterAlgorithm(
        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), obj.memoryLimitInBytes);
      backwardsFilterWorkSpace = gpu.allocateBackwardFilterWorkspace(gpu,
        inputDescriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), backwardFilterAlgorithm);
      backwardsDataWorkSpace = gpu.allocateBackwardDataWorkspace(gpu,
        inputDescriptor.getPtr(), filterDescriptor.getPtr(),
        convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), backwardDataAlgorithm);
    }
  
    /**
     * Free.
     */
    @Override
    public void _free() {
      this.convolutionDescriptor.freeRef();
      this.filterDescriptor.freeRef();
      this.inputDescriptor.freeRef();
      this.outputDescriptor.freeRef();
      this.backwardsFilterWorkSpace.freeRef();
      this.backwardsDataWorkSpace.freeRef();
      key.freeRef();
    }
  
    @Override
    public void detach() {
      this.convolutionDescriptor.detach();
      this.filterDescriptor.detach();
      this.inputDescriptor.detach();
      this.outputDescriptor.detach();
      this.backwardsFilterWorkSpace.detach();
      this.backwardsDataWorkSpace.detach();
      key.detach();
      super.detach();
    }
  }
  
  private static class CudaFwdParameters extends SimpleConvolutionParameters {
    /**
     * The Forward algorithm.
     */
    public final int forwardAlgorithm;
    /**
     * The Output dims.
     */
    public final int[] outputDims;
    /**
     * The Output descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnTensorDescriptor> outputDescriptor;
    /**
     * The Input descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnTensorDescriptor> inputDescriptor;
    /**
     * The Filter descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnFilterDescriptor> filterDescriptor;
    /**
     * The Convolution descriptor.
     */
    @javax.annotation.Nonnull
    public final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor;
    /**
     * The Forward workspace.
     */
    @javax.annotation.Nonnull
    public final CudaMemory forwardWorkspace;
    /**
     * The Key.
     */
    public final SimpleConvolutionParameters key;
  
    /**
     * Instantiates a new Cuda fwd parameters.
     *
     * @param obj the obj
     */
    CudaFwdParameters(@javax.annotation.Nonnull SimpleConvolutionParameters obj) {
      super(obj.kernel, obj.paddingX, obj.paddingY, obj.precision, obj.strideX, obj.strideY, obj.length, obj.inputSize, obj.outputSize, obj.kernelSize, obj.gpu, obj.memoryLimitInBytes);
      this.key = obj;
      key.addRef();
      inputDescriptor = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      filterDescriptor = gpu.newFilterDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      convolutionDescriptor = gpu.newConvolutions2dDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision.code,
        paddingY, paddingX,
        strideY, strideX,
        1, 1);
      outputDims = IntStream.of(reverse(CudaSystem.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr()))).limit(3).toArray();
      outputDescriptor = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, outputDims[2], outputDims[1], outputDims[0]);
      forwardAlgorithm = gpu.getForwardAlgorithm(
        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), obj.memoryLimitInBytes);
      forwardWorkspace = gpu.allocateForwardWorkspace(gpu,
        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), forwardAlgorithm);
    }
  
    /**
     * Free.
     */
    @Override
    public void _free() {
      super._free();
      this.convolutionDescriptor.freeRef();
      this.filterDescriptor.freeRef();
      this.inputDescriptor.freeRef();
      this.outputDescriptor.freeRef();
      this.forwardWorkspace.freeRef();
      key.freeRef();
    }
  
    @Override
    public void detach() {
      this.convolutionDescriptor.detach();
      this.filterDescriptor.detach();
      this.inputDescriptor.detach();
      this.outputDescriptor.detach();
      this.forwardWorkspace.detach();
      key.detach();
      super.detach();
    }
  }
}
