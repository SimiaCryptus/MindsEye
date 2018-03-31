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

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.util.lang.TimedResult;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnNanPropagation;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.function.Function;

/**
 * The type Gpu device.
 */
public class CudaDevice extends CudaSystem {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudnnHandle.class);
  private static final Object memoryManagementLock = new Object();
  /**
   * The Device name.
   */
  @Nullable
  protected final String deviceName;
  /**
   * The Device number.
   */
  protected final int deviceId;
  private volatile cudaDeviceProp deviceProperties;
  
  /**
   * Instantiates a new Gpu device.
   *
   * @param deviceId the device number
   */
  protected CudaDevice(final int deviceId) {
    super();
    this.deviceId = deviceId;
    assert 0 <= this.deviceId;
    if (0 <= this.deviceId) {
      initThread();
      deviceName = getDeviceName(deviceId);
    }
    else {
      deviceName = null;
    }
  }
  
  /**
   * Cuda freeRef int.
   *
   * @param deviceId the device id
   * @param devPtr   the dev ptr
   * @return the int
   */
  public static int cudaFree(int deviceId, final CudaPointer devPtr) {
    long startTime = System.nanoTime();
    if (null == devPtr) return 0;
    Function<CudnnHandle, Integer> fn = dev -> {
      final int result = JCuda.cudaFree(devPtr);
      log("cudaFree", result, new Object[]{devPtr});
      cudaFree_execution.accept((System.nanoTime() - startTime) / 1e9);
      handle(result);
      return result;
    };
    if (deviceId < 0) {
      return fn.apply(null);
    }
    else {
      return CudaSystem.withDevice(deviceId, fn);
    }
  }
  
  /**
   * Gets device name.
   *
   * @param device the device
   * @return the device name
   */
  public static String getDeviceName(final int device) {
    return new String(CudaDevice.getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }
  
  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  public static cudaDeviceProp getDeviceProperties(final int device) {
    return propertyCache.computeIfAbsent(device, deviceId -> {
      long startTime = System.nanoTime();
      @Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((System.nanoTime() - startTime) / 1e9);
      log("cudaGetDeviceProperties", result, new Object[]{deviceProp, device});
      return deviceProp;
    });
  }
  
  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(final int cudaDeviceId) {
    if (cudaDeviceId < 0) throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
    if (!isThreadDeviceId(cudaDeviceId)) {
      long startTime = System.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((System.nanoTime() - startTime) / 1e9);
      log("cudaSetDevice", result, new Object[]{cudaDeviceId});
      CudaSystem.handle(result);
      CudaSystem.currentDeviceId.set(cudaDeviceId);
    }
  }
  
  /**
   * Acquire pointer.
   *
   * @param size    the size
   * @param type    the type
   * @param retries the retries
   * @return the pointer
   */
  @Nonnull
  CudaPointer acquire(long size, @Nonnull MemoryType type, int retries) {
    assert CudaSystem.isThreadDeviceId(getDeviceId());
    if (retries < 0) throw new IllegalArgumentException();
    final DeviceMetrics metrics = ensureCapacity(size, type);
    try {
      @Nonnull CudaPointer pointer = type.allocCached(size, this);
      final long finalMemory = metrics.activeMemory.addAndGet(size);
      metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      return pointer;
    } catch (@Nonnull final ThreadDeath e) {
      throw e;
    } catch (@Nonnull final Throwable e) {
      if (retries <= 0)
        throw new RuntimeException(String.format(String.format("Error allocating %e bytes; %s currently allocated to device %s", (double) size, metrics.usedMemory, this)), e);
      final long startMemory = metrics.usedMemory.get();
      @Nonnull TimedResult<Double> timedResult = TimedResult.time(() -> CudaMemory.clearMemory(getDeviceId()));
      final long freedMemory = startMemory - metrics.usedMemory.get();
      CudaMemory.logger.warn(String.format("Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)",
        size, freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
    }
    if (retries < 0) throw new IllegalStateException();
    return this.acquire(size, type, retries - 1);
  }
  
  /**
   * Ensure capacity device metrics.
   *
   * @param size the size
   * @param type the type
   * @return the device metrics
   */
  @Nonnull
  public DeviceMetrics ensureCapacity(final long size, final MemoryType type) {
    final DeviceMetrics metrics;
    synchronized (memoryManagementLock) {
      if (size <= 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      metrics = CudaMemory.getGpuStats(deviceId);
      double resultingTotalMemory = CudaMemory.METRICS.values().stream().mapToLong(m -> m.usedMemory.get()).sum() + size;
      if (resultingTotalMemory > CudaSettings.INSTANCE.getMaxTotalMemory()) {
        CudaMemory.logger.info(String.format("Clearing weak global memory while allocating %e bytes (%e > %e)", (double) size, resultingTotalMemory, CudaSettings.INSTANCE.getMaxTotalMemory()));
        CudaMemory.clearWeakMemory(deviceId);
      }
      resultingTotalMemory = CudaMemory.METRICS.values().stream().mapToLong(x1 -> x1.usedMemory.get()).sum() + size;
      if (resultingTotalMemory > CudaSettings.INSTANCE.getMaxTotalMemory()) {
        CudaMemory.logger.info(String.format("Clearing all global memory while allocating %e bytes (%e > %e)", (double) size, resultingTotalMemory, CudaSettings.INSTANCE.getMaxTotalMemory()));
        CudaMemory.clearMemory(deviceId);
      }
      double resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > CudaSettings.INSTANCE.getMaxDeviceMemory()) {
        CudaMemory.logger.info(String.format("Clearing weak memory for device %s while allocating %e bytes (%e > %e)", this, (double) size, resultingDeviceMemory, CudaSettings.INSTANCE.getMaxDeviceMemory()));
        CudaMemory.METRICS.keySet().stream().mapToInt(x -> x).distinct().forEach(CudaMemory::clearWeakMemory);
      }
      resultingDeviceMemory = metrics.usedMemory.get() + size;
      if (resultingDeviceMemory > CudaSettings.INSTANCE.getMaxDeviceMemory()) {
        CudaMemory.logger.info(String.format("Clearing all memory for device %s while allocating %e bytes (%s > %e)", this, (double) size, resultingDeviceMemory, CudaSettings.INSTANCE.getMaxDeviceMemory()));
        CudaMemory.METRICS.keySet().stream().mapToInt(x -> x).distinct().forEach(CudaMemory::clearMemory);
      }
    }
    return metrics;
  }
  
  /**
   * New convolution nd descriptor cuda resource.
   *
   * @param mode     the mode
   * @param dataType the data type
   * @param padding  the padding
   * @param stride   the stride
   * @param dilation the dilation
   * @return the cuda resource
   */
  public CudaResource<cudnnConvolutionDescriptor> newConvolutionNdDescriptor(final int mode, final int dataType, @Nonnull final int[] padding, @Nonnull final int[] stride, @Nonnull final int[] dilation) {
    long startTime = System.nanoTime();
    assert padding.length == stride.length;
    assert padding.length == dilation.length;
    assert Arrays.stream(padding).allMatch(x -> x >= 0);
    assert Arrays.stream(stride).allMatch(x -> x > 0);
    assert Arrays.stream(dilation).allMatch(x -> x > 0);
    @Nonnull final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    newConvolutionNdDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnCreateConvolutionDescriptor", result, new Object[]{convDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc,
      3,
      padding,
      stride,
      dilation,
      mode,
      dataType
    );
    log("cudnnSetConvolutionNdDescriptor", result, new Object[]{convDesc, padding.length, padding, stride, dilation, mode, dataType});
    CudaSystem.handle(result);
    return new CudaResource<cudnnConvolutionDescriptor>(convDesc, CudaSystem::cudnnDestroyConvolutionDescriptor, getDeviceId()) {
      @Nonnull
      @Override
      public String toString() {
        return "cudnnSetConvolutionNdDescriptor(padding=" + Arrays.toString(padding) +
          ";stride=" + Arrays.toString(stride) +
          ";dilation=" + Arrays.toString(dilation) +
          ";mode=" + mode +
          ";dataType=" + dataType + ")";
      }
    };
  }
  
  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType     the data type
   * @param tensorLayout the tensor layout
   * @param dimensions   the dimensions
   * @return the cuda resource
   */
  public CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final int dataType, final int tensorLayout, @Nonnull final int[] dimensions) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, new Object[]{filterDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetFilterNdDescriptor", result, new Object[]{filterDesc, dataType, tensorLayout, dimensions.length, dimensions});
    CudaSystem.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CudaSystem::cudnnDestroyFilterDescriptor, getDeviceId()) {
      @Nonnull
      @Override
      public String toString() {
        return "cudnnSetFilterNdDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";dimensions=" + Arrays.toString(dimensions) + ")";
      }
    };
  }
  
  /**
   * Allocate cuda ptr.
   *
   * @param size  the size
   * @param type  the type
   * @param dirty the dirty
   * @return the cuda ptr
   */
  @Nonnull
  public CudaMemory allocate(final long size, @Nonnull MemoryType type, boolean dirty) {
    assert CudaSystem.isThreadDeviceId(getDeviceId());
    @Nonnull CudaMemory obtain = new CudaMemory(this, size, type);
    if (!dirty) obtain.clear();
    return obtain;
  }
  
  
  /**
   * New tensor descriptor cuda resource.
   *
   * @param dataType   the data type
   * @param batchCount the batch count
   * @param channels   the channels
   * @param height     the height
   * @param width      the width
   * @return the cuda resource
   */
  public CudaTensorDescriptor newTensorDescriptor(final Precision dataType,
    final int batchCount, final int channels, final int height, final int width) {
    return newTensorDescriptor(dataType, batchCount, channels, height, width, channels * height * width, height * width, width, 1);
  }
  
  /**
   * New tensor descriptor cuda resource.
   *
   * @param dataType   the data type
   * @param batchCount the batch count
   * @param channels   the channels
   * @param height     the height
   * @param width      the width
   * @param nStride    the n stride
   * @param cStride    the c stride
   * @param hStride    the h stride
   * @param wStride    the w stride
   * @return the cuda resource
   */
  public CudaTensorDescriptor newTensorDescriptor(final Precision dataType,
    final int batchCount, final int channels, final int height, final int width,
    final int nStride, final int cStride, final int hStride, final int wStride) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    log("cudnnCreateTensorDescriptor", result, new Object[]{desc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType.code, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetTensor4dDescriptorEx", result, new Object[]{desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride});
    CudaSystem.handle(result);
    return new CudaTensorDescriptor(desc, getDeviceId(), dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
  }
  
  /**
   * New op descriptor cuda resource.
   *
   * @param opType   the op type
   * @param dataType the data type
   * @return the cuda resource
   */
  public CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(final int opType, final Precision dataType) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    log("cudnnCreateOpTensorDescriptor", result, new Object[]{opDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetOpTensorDescriptor", result, new Object[]{opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN});
    CudaSystem.handle(result);
    return new CudaResource<>(opDesc, CudaSystem::cudnnDestroyOpTensorDescriptor, getDeviceId());
  }
  
  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType       the data type
   * @param tensorLayout   the tensor layout
   * @param outputChannels the output channels
   * @param inputChannels  the input channels
   * @param height         the height
   * @param width          the width
   * @return the cuda resource
   */
  public CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final Precision dataType, final int tensorLayout, final int outputChannels, final int inputChannels, final int height, final int width) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, new Object[]{filterDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType.code, tensorLayout, outputChannels, inputChannels, height, width);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetFilter4dDescriptor", result, new Object[]{filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width});
    CudaSystem.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CudaSystem::cudnnDestroyFilterDescriptor, getDeviceId()) {
      @Nonnull
      @Override
      public String toString() {
        return "cudnnSetFilter4dDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";outputChannels=" + outputChannels +
          ";inputChannels=" + inputChannels +
          ";height=" + height +
          ";=width" + width + ")";
      }
    };
  }
  
  /**
   * New convolutions 2 d descriptor cuda resource.
   *
   * @param mode         the mode
   * @param dataType     the data type
   * @param paddingY     the padding y
   * @param paddingX     the padding x
   * @param strideHeight the stride height
   * @param strideWidth  the stride width
   * @param dilationY    the dilation y
   * @param dilationX    the dilation x
   * @return the cuda resource
   */
  public CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int mode, final Precision dataType, final int paddingY, final int paddingX, final int strideHeight, final int strideWidth, int dilationY, int dilationX) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    log("cudnnCreateConvolutionDescriptor", result, new Object[]{convDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetConvolution2dDescriptor(
      convDesc,
      paddingY, // zero-padding height
      paddingX, // zero-padding width
      strideHeight, // vertical filter stride
      strideWidth, // horizontal filter stride
      dilationY, // upscale the input in x-direction
      dilationX, // upscale the input in y-direction
      mode
      , dataType.code
    );
    newConvolutions2dDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetConvolution2dDescriptor", result, new Object[]{convDesc, paddingY, paddingX, strideHeight, strideWidth, dilationY, dilationX, mode, dataType});
    CudaSystem.handle(result);
    return new CudaResource<>(convDesc, CudaSystem::cudnnDestroyConvolutionDescriptor, getDeviceId());
  }
  
  /**
   * New activation descriptor cuda resource.
   *
   * @param mode     the mode
   * @param reluNan  the relu nan
   * @param reluCeil the relu ceil
   * @return the cuda resource
   */
  public CudaResource<cudnnActivationDescriptor> newActivationDescriptor(final int mode, final int reluNan, final double reluCeil) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    log("cudnnCreateActivationDescriptor", result, new Object[]{desc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnSetActivationDescriptor", result, new Object[]{desc, mode, reluNan, reluCeil});
    CudaSystem.handle(result);
    return new CudaResource<>(desc, CudaSystem::cudnnDestroyActivationDescriptor, getDeviceId());
  }
  
  /**
   * Create pooling descriptor cuda resource.
   *
   * @param mode       the mode
   * @param poolDims   the pool dims
   * @param windowSize the window size
   * @param padding    the padding
   * @param stride     the stride
   * @return the cuda resource
   */
  public CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(final int mode, final int poolDims, final int[] windowSize, final int[] padding, final int[] stride) {
    long startTime = System.nanoTime();
    @Nonnull final cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    log("cudnnCreatePoolingDescriptor", result, new Object[]{poolingDesc});
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc,
      mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
      padding, stride);
    log("cudnnSetPoolingNdDescriptor", result, new Object[]{poolingDesc, mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize, padding, stride});
    CudaSystem.handle(result);
    createPoolingDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, CudaSystem::cudnnDestroyPoolingDescriptor, getDeviceId());
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    setDevice(getDeviceId());
  }
  
  /**
   * Gets device properties.
   *
   * @return the device properties
   */
  public cudaDeviceProp getDeviceProperties() {
    if (null == deviceProperties) {
      deviceProperties = getDeviceProperties(getDeviceId());
    }
    return deviceProperties;
  }
  
  /**
   * Gets device number.
   *
   * @return the device number
   */
  public int getDeviceId() {
    return deviceId;
  }
  
  /**
   * Gets device name.
   *
   * @return the device name
   */
  @Nonnull
  public CharSequence getDeviceName() {
    return new String(getDeviceProperties().name, Charset.forName("ASCII")).trim();
  }
  
  /**
   * The type Cuda tensor descriptor.
   */
  public static class CudaTensorDescriptor extends CudaResource<cudnnTensorDescriptor> {
  
    /**
     * The W stride.
     */
    public final int wStride;
    /**
     * The H stride.
     */
    public final int hStride;
    /**
     * The C stride.
     */
    public final int cStride;
    /**
     * The N stride.
     */
    public final int nStride;
    /**
     * The Width.
     */
    public final int width;
    /**
     * The Height.
     */
    public final int height;
    /**
     * The Channels.
     */
    public final int channels;
    /**
     * The Batch count.
     */
    public final int batchCount;
    /**
     * The Data type.
     */
    public final Precision dataType;
  
    /**
     * Instantiates a new Cuda resource.
     *
     * @param obj        the obj
     * @param deviceId   the device id
     * @param dataType   the data type
     * @param batchCount the batch count
     * @param channels   the channels
     * @param height     the height
     * @param width      the width
     * @param nStride    the n stride
     * @param cStride    the c stride
     * @param hStride    the h stride
     * @param wStride    the w stride
     */
    protected CudaTensorDescriptor(final cudnnTensorDescriptor obj, final int deviceId, final Precision dataType,
      final int batchCount, final int channels, final int height, final int width,
      final int nStride, final int cStride, final int hStride, final int wStride) {
      super(obj, CudaSystem::cudnnDestroyTensorDescriptor, deviceId);
      this.dataType = dataType;
      this.batchCount = batchCount;
      this.channels = channels;
      this.height = height;
      this.width = width;
      this.nStride = nStride;
      this.cStride = cStride;
      this.hStride = hStride;
      this.wStride = wStride;
    }
  
    /**
     * Copy cuda tensor descriptor.
     *
     * @param device the device
     * @return the cuda tensor descriptor
     */
    public CudaTensorDescriptor copy(CudaDevice device) {
      return device.newTensorDescriptor(dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    }
  }
}
