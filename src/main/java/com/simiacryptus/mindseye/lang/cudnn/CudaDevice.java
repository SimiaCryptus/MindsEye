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

import com.simiacryptus.mindseye.lang.ReshapedTensorList;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.lang.TimedResult;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.charset.Charset;
import java.util.Arrays;

/**
 * The type Gpu device.
 */
public class CudaDevice extends CudaSystem {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudnnHandle.class);
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
  public CudaDevice(final int deviceId) {
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
  public static int cudaFree(int deviceId, final Pointer devPtr) {
    long startTime = System.nanoTime();
    return CudaSystem.withDevice(deviceId, () -> {
      final int result = JCuda.cudaFree(devPtr);
      CudaSystem.log("cudaFree", result, devPtr);
      cudaFree_execution.accept((System.nanoTime() - startTime) / 1e9);
      handle(result);
      return result;
    });
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
      @javax.annotation.Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((System.nanoTime() - startTime) / 1e9);
      CudaSystem.log("cudaGetDeviceProperties", result, deviceProp, device);
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
    if (cudaDeviceId != getThreadDevice()) {
      long startTime = System.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((System.nanoTime() - startTime) / 1e9);
      CudaSystem.log("cudaSetDevice", result, cudaDeviceId);
      CudaSystem.handle(result);
      CudaSystem.currentDevice.set(cudaDeviceId);
    }
  }
  
  /**
   * The Ptr.
   *
   * @param data       the data
   * @param memoryType the memory type
   * @return the ptr
   */
  @Nonnull
  public CudaMemory getPtr(@Nonnull final CudaTensorList data, @Nonnull final MemoryType memoryType) {
    CudaMemory ptr = data.ptr;
    synchronized (data) {
      if ((null == ptr || data.ptr.isFinalized()) && null != data.heapCopy && !data.heapCopy.isFinalized()) {
        synchronized (data) {
          if ((null == data.ptr || data.ptr.isFinalized()) && null != data.heapCopy && !data.heapCopy.isFinalized()) {
            data.ptr = ptr = getPtr(data.heapCopy, data.precision, memoryType);
          }
        }
      }
    }
    if (null == ptr) {
      if (null == data.heapCopy) {
        throw new IllegalStateException("No data");
      }
      else if (data.heapCopy.isFinalized()) {
        throw new IllegalStateException("Local data has been freed");
      }
    }
    ptr.addRef();
    return ptr.moveTo(this, memoryType);
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
  Pointer acquire(long size, @Nonnull MemoryType type, int retries) {
    if (retries < 0) throw new IllegalArgumentException();
    if (size <= 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    final DeviceMetrics metrics = CudaMemory.getGpuStats(getDeviceId());
    long totalGpuMem = CudaMemory.METRICS.asMap().values().stream().mapToLong(x -> x.usedMemory.get()).sum();
    long resultingTotalMemory = totalGpuMem + size;
    long resultingDeviceMemory = metrics.usedMemory.get() + size;
    if (resultingDeviceMemory > metrics.highMemoryThreshold || resultingTotalMemory > CudaSettings.INSTANCE.getMaxTotalMemory()) {
      CudaMemory.logger.info(String.format("Clearing memory for device %s while allocating %s bytes (%s > %s)", this, size, resultingDeviceMemory, metrics.highMemoryThreshold));
      CudaMemory.clearMemory(getDeviceId());
    }
    try {
      @Nonnull Pointer pointer = new Pointer();
      type.alloc(size, pointer, this);
      final long finalMemory = metrics.usedMemory.addAndGet(size);
      metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      return pointer;
    } catch (@Nonnull final ThreadDeath e) {
      throw e;
    } catch (@Nonnull final Throwable e) {
      if (retries <= 0)
        throw new RuntimeException(String.format(String.format("Error allocating %d bytes; %s currently allocated to device %s", size, metrics.usedMemory, this)), e);
      final long startMemory = metrics.usedMemory.get();
      @Nonnull TimedResult<Void> timedResult = TimedResult.time(() -> CudaMemory.clearMemory(getDeviceId()));
      final long freedMemory = startMemory - metrics.usedMemory.get();
      CudaMemory.logger.warn(String.format("Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)",
        size, freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
    }
    if (retries < 0) throw new IllegalStateException();
    return this.acquire(size, type, retries - 1);
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
    CudaSystem.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc,
      3,
      padding,
      stride,
      dilation,
      mode,
      dataType
    );
    CudaSystem.log("cudnnSetConvolutionNdDescriptor", result, convDesc, padding.length,
      padding,
      stride,
      dilation,
      mode,
      dataType);
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
    CudaSystem.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetFilterNdDescriptor", result, filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
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
    @Nonnull CudaMemory obtain = new CudaMemory(size, this, type);
    if (!dirty) obtain.clear();
    return obtain;
  }
  
  /**
   * Gets cuda ptr.
   *
   * @param data       the data
   * @param precision  the precision
   * @param memoryType the memory type
   * @return the cuda ptr
   */
  @Nonnull
  public CudaMemory getPtr(@Nonnull final TensorList data, @Nonnull final Precision precision, final MemoryType memoryType) {
    data.assertAlive();
    if (data instanceof ReshapedTensorList) {
      return getPtr(((ReshapedTensorList) data).getInner(), precision, memoryType);
    }
    if (data instanceof CudaTensorList && precision == ((CudaTensorList) data).getPrecision()) {
      @Nonnull CudaTensorList cudaTensorList = (CudaTensorList) data;
      return this.getPtr(cudaTensorList, memoryType);
    }
    else {
      final int listLength = data.length();
      final int elementLength = Tensor.dim(data.getDimensions());
      @Nonnull final CudaMemory ptr = this.allocate((long) elementLength * listLength * precision.size, memoryType, true);
      for (int i = 0; i < listLength; i++) {
        Tensor tensor = data.get(i);
        assert null != data;
        assert null != tensor;
        assert Arrays.equals(tensor.getDimensions(), data.getDimensions()) : Arrays.toString(tensor.getDimensions()) + " != " + Arrays.toString(data.getDimensions());
        ptr.write(precision, tensor.getData(), (long) i * elementLength);
        tensor.freeRef();
      }
      return ptr;
    }
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
  public CudaResource<cudnnTensorDescriptor> newTensorDescriptor(final int dataType,
    final int batchCount, final int channels, final int height, final int width,
    final int nStride, final int cStride, final int hStride, final int wStride) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    CudaSystem.log("cudnnCreateTensorDescriptor", result, desc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetTensor4dDescriptorEx", result, desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    CudaSystem.handle(result);
    return new CudaResource<>(desc, CudaSystem::cudnnDestroyTensorDescriptor, getDeviceId());
  }
  
  /**
   * New tensor descriptor cuda resource.
   *
   * @param dataType     the data type
   * @param tensorLayout the tensor layout
   * @param batchCount   the batch count
   * @param channels     the channels
   * @param height       the height
   * @param width        the width
   * @return the cuda resource
   */
  public CudaResource<cudnnTensorDescriptor> newTensorDescriptor(final int dataType, final int tensorLayout,
    final int batchCount, final int channels, final int height, final int width) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    CudaSystem.log("cudnnCreateTensorDescriptor", result, desc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetTensor4dDescriptor", result, desc, tensorLayout, dataType, batchCount, channels, height, width);
    CudaSystem.handle(result);
    return new CudaResource<cudnnTensorDescriptor>(desc, CudaSystem::cudnnDestroyTensorDescriptor, getDeviceId()) {
      @javax.annotation.Nonnull
      @Override
      public String toString() {
        return "cudnnSetTensor4dDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";batchCount=" + batchCount +
          ";channels=" + channels +
          ";height=" + height +
          ";=width" + width + ")";
      }
    };
  }
  
  /**
   * New op descriptor cuda resource.
   *
   * @param opType   the op type
   * @param dataType the data type
   * @return the cuda resource
   */
  public CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(final int opType, final int dataType) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    CudaSystem.log("cudnnCreateOpTensorDescriptor", result, opDesc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetOpTensorDescriptor", result, opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
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
  public CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final int dataType, final int tensorLayout, final int outputChannels, final int inputChannels, final int height, final int width) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    CudaSystem.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetFilter4dDescriptor", result, filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    CudaSystem.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CudaSystem::cudnnDestroyFilterDescriptor, getDeviceId()) {
      @javax.annotation.Nonnull
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
  public CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int mode, final int dataType, final int paddingY, final int paddingX, final int strideHeight, final int strideWidth, int dilationY, int dilationX) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    CudaSystem.log("cudnnCreateConvolutionDescriptor", result, convDesc);
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
      , dataType
    );
    newConvolutions2dDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetConvolution2dDescriptor", result, convDesc,
      paddingY, // zero-padding height
      paddingX, // zero-padding width
      strideHeight, // vertical filter stride
      strideWidth, // horizontal filter stride
      dilationY, // upscale the input in x-direction
      dilationX, // upscale the input in y-direction
      mode,
      dataType);
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
    @javax.annotation.Nonnull final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    CudaSystem.log("cudnnCreateActivationDescriptor", result, desc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnSetActivationDescriptor", result, desc, mode, reluNan, reluCeil);
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
    @javax.annotation.Nonnull final cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    CudaSystem.log("cudnnCreatePoolingDescriptor", result, poolingDesc);
    CudaSystem.handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc,
      mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
      padding, stride);
    CudaSystem.log("cudnnSetPoolingNdDescriptor", result, poolingDesc,
      mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
      padding, stride);
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
      synchronized (this) {
        if (null == deviceProperties) {
          deviceProperties = getDeviceProperties(getDeviceId());
        }
      }
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
  @javax.annotation.Nonnull
  public String getDeviceName() {
    return new String(getDeviceProperties().name, Charset.forName("ASCII")).trim();
  }
}
