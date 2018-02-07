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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.lang.TimedResult;
import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutionException;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

/**
 * A GPU memory segment
 */
public class CudaPtr extends CudaResourceBase<Pointer> {
  /**
   * The constant METRICS.
   */
  public static final LoadingCache<Integer, GpuStats> METRICS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, GpuStats>() {
    @Override
    public @NotNull GpuStats load(final Integer integer) throws Exception {
      return new GpuStats();
    }
  });
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaPtr.class);
  /**
   * The Max.
   */
  static final long MAX = Precision.Double.size * (Integer.MAX_VALUE - 1L);
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  /**
   * The Size.
   */
  public final long size;
  private final int deviceId;
  private final @NotNull MemoryType type;
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   */
  private CudaPtr(final long size, final int deviceId, @NotNull MemoryType type) {
    super(acquire(deviceId, size, type, 1));
    this.size = size;
    this.deviceId = deviceId;
    this.type = type;
  }
  
  /**
   * Allocate cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @param type     the type
   * @param dirty    the dirty
   * @return the cuda ptr
   */
  public static @NotNull CudaPtr allocate(final int deviceId, final long size, @NotNull MemoryType type, boolean dirty) {
    @NotNull CudaPtr obtain = new CudaPtr(size, type == MemoryType.Device ? deviceId : -1, type);
    if (!dirty) obtain.clear();
    return obtain;
  }
  
  /**
   * Gets cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public static CudaPtr getCudaPtr(final @NotNull Precision precision, final @NotNull TensorList data) {
    data.assertAlive();
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision() && ((GpuTensorList) data).isNative()) {
      @NotNull GpuTensorList gpuTensorList = (GpuTensorList) data;
      final CudaPtr ptr = gpuTensorList.getPtr();
      assert null != ptr;
      ptr.addRef();
      return ptr;
    }
    else {
      final int listLength = data.length();
      final int elementLength = Tensor.dim(data.getDimensions());
      final double[] inputBuffer = RecycleBin.DOUBLES.obtain(elementLength * listLength);
      for (int i = 0; i < listLength; i++) {
        Tensor tensor = data.get(i);
        final double[] doubles = tensor.getData();
        assert elementLength == doubles.length;
        System.arraycopy(doubles, 0, inputBuffer, i * elementLength, elementLength);
        tensor.freeRef();
      }
      final @NotNull CudaPtr ptr = CudaPtr.allocate(GpuSystem.getDevice(), (long) inputBuffer.length * precision.size, MemoryType.Managed, true).write(precision, inputBuffer);
      RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
      return ptr;
    }
  }
  
  private static @NotNull Pointer acquire(int deviceId, long size, @NotNull MemoryType type, int retries) {
    if (retries < 0) throw new IllegalArgumentException();
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaPtr.MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (deviceId >= 0 && GpuSystem.getDevice() != deviceId) throw new IllegalArgumentException();
    final GpuStats metrics = CudaPtr.getGpuStats(deviceId);
    try {
      @NotNull Pointer pointer = new Pointer();
      type.alloc(size, pointer);
      final long finalMemory = metrics.usedMemory.addAndGet(size);
      metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      return pointer;
    } catch (final @NotNull ThreadDeath e) {
      throw e;
    } catch (final @NotNull Throwable e) {
      if (retries <= 0) throw new RuntimeException(e);
      final long startMemory = metrics.usedMemory.get();
      @NotNull TimedResult<Void> timedResult = TimedResult.time(() -> GpuSystem.cleanMemory());
      final long freedMemory = startMemory - metrics.usedMemory.get();
      logger.warn(String.format("Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)",
                                size, freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
    }
    if (retries < 0) throw new IllegalStateException();
    return acquire(deviceId, size, type, retries - 1);
  }
  
  /**
   * From device double tensor.
   *
   * @param ptr        the filter data
   * @param precision  the precision
   * @param dimensions the dimensions  @return the tensor
   * @return the tensor
   */
  public static @NotNull Tensor read(final @NotNull CudaPtr ptr, final @NotNull Precision precision, final int[] dimensions) {
    GpuSystem.cudaDeviceSynchronize();
    final @NotNull Tensor tensor = new Tensor(dimensions);
    switch (precision) {
      case Float:
        final int length = tensor.dim();
        final @NotNull float[] data = new float[length];
        ptr.read(precision, data);
        final double[] doubles = tensor.getData();
        for (int i = 0; i < length; i++) {
          doubles[i] = data[i];
        }
        break;
      case Double:
        ptr.read(precision, tensor.getData());
        break;
      default:
        throw new IllegalStateException();
    }
    return tensor;
  }
  
  
  /**
   * Gets current device properties.
   *
   * @return the current device properties
   */
  static cudaDeviceProp getCurrentDeviceProperties() {
    return GpuDevice.getDeviceProperties(GpuSystem.getDevice());
  }
  
  /**
   * Gets gpu stats.
   *
   * @param deviceId the device id
   * @return the gpu stats
   */
  public static GpuStats getGpuStats(final int deviceId) {
    GpuStats devivceMemCtr;
    try {
      devivceMemCtr = CudaPtr.METRICS.get(deviceId);
    } catch (final @NotNull ExecutionException e) {
      throw new RuntimeException(e.getCause());
    }
    return devivceMemCtr;
  }
  
  /**
   * Copy to cuda ptr.
   *
   * @param deviceId the device id
   * @return the cuda ptr
   */
  public CudaPtr copyTo(int deviceId) {
    return GpuSystem.withDevice(deviceId, () -> {
      @NotNull CudaPtr copy = allocate(deviceId, size, MemoryType.Managed, false);
      GpuSystem.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
      return copy;
    });
  }
  
  /**
   * Move to cuda ptr.
   *
   * @param deviceId the device id
   * @return the cuda ptr
   */
  public CudaPtr moveTo(int deviceId) {
    if (deviceId == getDeviceId()) return this;
    else return copyTo(deviceId);
  }
  
  @Override
  protected void _free() {
    if (isActiveObj()) {
      getType().free(ptr, deviceId);
      CudaPtr.getGpuStats(deviceId).usedMemory.addAndGet(-size);
    }
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @return the cuda ptr
   */
  public @NotNull CudaPtr read(final @NotNull Precision precision, final @NotNull double[] destination) {
    if (size != destination.length * 1l * precision.size) {
      throw new IllegalArgumentException(size + " != " + destination.length * 1l * precision.size);
    }
    if (precision == Precision.Float) {
      @NotNull float[] data = new float[destination.length];
      read(Precision.Float, data);
      for (int i = 0; i < data.length; i++) {
        destination[i] = data[i];
      }
    }
    else {
      GpuSystem.cudaMemcpy(precision.getPointer(destination), getPtr(), size, cudaMemcpyDeviceToHost);
      CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
    }
    return this;
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @return the cuda ptr
   */
  public @NotNull CudaPtr read(final @NotNull Precision precision, final @NotNull float[] destination) {
    if (size != destination.length * 1l * precision.size) {
      throw new IllegalArgumentException(size + " != " + destination.length * 1l * precision.size);
    }
    if (precision == Precision.Double) {
      @NotNull double[] data = new double[destination.length];
      read(Precision.Double, data);
      for (int i = 0; i < data.length; i++) {
        destination[i] = (float) data[i];
      }
    }
    else {
      GpuSystem.cudaMemcpy(precision.getPointer(destination), getPtr(), size, cudaMemcpyDeviceToHost);
      CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
    }
    return this;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public @NotNull CudaPtr write(final @NotNull Precision precision, final @NotNull double[] data) {
    if (size != (long) data.length * precision.size)
      throw new IllegalArgumentException(String.format("%d != %d * %d", size, data.length, precision.size));
    final @NotNull Pointer src = precision.getPointer(data);
    GpuSystem.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaPtr.getGpuStats(deviceId).memoryWrites.addAndGet(size);
    return this;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public @NotNull CudaPtr write(final @NotNull Precision precision, final @NotNull float[] data) {
    if (size != data.length * precision.size) throw new IllegalArgumentException();
    final @NotNull Pointer src = precision.getPointer(data);
    GpuSystem.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaPtr.getGpuStats(deviceId).memoryWrites.addAndGet(size);
    return this;
  }
  
  /**
   * Gets device id.
   *
   * @return the device id
   */
  public int getDeviceId() {
    return deviceId;
  }
  
  /**
   * Gets type.
   *
   * @return the type
   */
  public @NotNull MemoryType getType() {
    return type;
  }
  
  private @NotNull CudaPtr clear() {
    GpuSystem.cudaMemset(getPtr(), 0, size);
    return this;
  }
  
  /**
   * As copy cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr asCopy() {
    CudaPtr copy = copy();
    freeRef();
    return copy;
  }
  
  /**
   * Copy cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr copy() {
    return copyTo(getDeviceId());
  }
}
