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
import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.RegisteredObjectBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer;
import jcuda.Pointer;
import jcuda.runtime.cudaMemcpyKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

/**
 * A GPU memory segment
 */
public class CudaMemory extends CudaResourceBase<Pointer> {
  /**
   * The constant METRICS.
   */
  public static final LoadingCache<Integer, DeviceMetrics> METRICS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, DeviceMetrics>() {
    @javax.annotation.Nonnull
    @Override
    public DeviceMetrics load(final Integer integer) throws Exception {
      return new DeviceMetrics();
    }
  });
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaMemory.class);
  /**
   * The K.
   */
  static final int K = 1024;
  /**
   * The Mi b.
   */
  static final int MiB = K * 1024;
  /**
   * The Gi b.
   */
  static final long GiB = 1024 * MiB;
  /**
   * The Size.
   */
  public final long size;
  private final int deviceId;
  @javax.annotation.Nonnull
  private final MemoryType type;
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   */
  CudaMemory(final long size, final CudaDevice deviceId, @javax.annotation.Nonnull MemoryType type) {
    super(deviceId.acquire(size, type, 1));
    this.size = size;
    this.deviceId = deviceId.getDeviceId();
    this.type = type;
  }
  
  /**
   * Clear memory.
   *
   * @param deviceId the device id
   */
  public static void clearMemory(final int deviceId) {
    if (CoreSettings.INSTANCE.isConservative()) {
      logLoad();
      logger.info(String.format("Running Garbage Collector"));
      System.gc();
    }
    long totalFreed = evictMemory(deviceId);
    if (totalFreed == 0) {
      logger.info(String.format("Nothing Freed - Running Garbage Collector"));
      System.gc();
      totalFreed = evictMemory(0);
    }
    if (totalFreed == 0) {
      logger.info(String.format("Warning: High Active GPU Memory Usage"));
    }
    logLoad();
  }
  
  /**
   * Evict memory long.
   *
   * @param deviceId the device id
   * @return the long
   */
  public static long evictMemory(final int deviceId) {
    logLoad();
    long bytes = RegisteredObjectBase.getLivingInstances(SimpleConvolutionLayer.class).mapToLong(x -> x.evictDeviceData(deviceId)).sum();
    logger.info(String.format("Cleared %s bytes from ConvolutionFilters for device %s", bytes, deviceId));
    long tensorListsFreed = CudaTensorList.evictToHeap(deviceId);
    return tensorListsFreed + bytes;
  }
  
  private static void logLoad() {
    logger.info(String.format("Current Load: %s", METRICS.asMap().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> {
      return String.format("%d bytes", e.getValue().usedMemory.get());
    }))));
  }
  
  
  /**
   * Gets gpu stats.
   *
   * @param deviceId the device id
   * @return the gpu stats
   */
  public static DeviceMetrics getGpuStats(final int deviceId) {
    DeviceMetrics devivceMemCtr;
    try {
      devivceMemCtr = CudaMemory.METRICS.get(deviceId);
    } catch (@javax.annotation.Nonnull final ExecutionException e) {
      throw new RuntimeException(e.getCause());
    }
    return devivceMemCtr;
  }
  
  /**
   * From device double tensor.
   *
   * @param precision  the precision
   * @param dimensions the dimensions  @return the tensor
   * @return the tensor
   */
  @Nonnull
  public Tensor read(@Nonnull final Precision precision, final int[] dimensions) {
    //CudaSystem.cudaDeviceSynchronize();
    @Nonnull final Tensor tensor = new Tensor(dimensions);
    switch (precision) {
      case Float:
        final int length = tensor.dim();
        @Nonnull final float[] data = new float[length];
        read(precision, data);
        @Nullable final double[] doubles = tensor.getData();
        for (int i = 0; i < length; i++) {
          doubles[i] = data[i];
        }
        break;
      case Double:
        read(precision, tensor.getData());
        break;
      default:
        throw new IllegalStateException();
    }
    return tensor;
  }
  
  /**
   * Copy to cuda ptr.
   *
   * @param deviceId   the device id
   * @param memoryType the memory type
   * @return the cuda ptr
   */
  public CudaMemory copyTo(CudaDevice deviceId, final MemoryType memoryType) {
    @javax.annotation.Nonnull CudaMemory copy = deviceId.allocate(size, memoryType, false);
    CudaSystem.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    return copy;
  }
  
  /**
   * Move to cuda ptr.
   *
   * @param deviceId   the device id
   * @param memoryType the memory type
   * @return the cuda ptr
   */
  public CudaMemory moveTo(CudaDevice deviceId, final MemoryType memoryType) {
    if (type == MemoryType.Managed) {
      return this;
    }
    else if (deviceId.getDeviceId() == getDeviceId()) {
      return this;
    }
    else {
      CudaMemory cudaMemory = copyTo(deviceId, memoryType);
      freeRef();
      return cudaMemory;
    }
  }
  
  
  /**
   * Free.
   */
  protected void _free() {
    CudnnHandle threadHandle = CudnnHandle.getThreadHandle();
    if (null != threadHandle) threadHandle.cleanupNative.add(this);
    else release();
  }
  
  @Override
  public void release() {
    if (isActiveObj()) {
      getType().free(ptr, deviceId);
      CudaMemory.getGpuStats(deviceId).usedMemory.addAndGet(-size);
    }
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @return the cuda ptr
   */
  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final double[] destination) {return read(precision, destination, 0);}
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @param offset      the offset
   * @return the cuda ptr
   */
  @javax.annotation.Nonnull
  public CudaMemory read(@javax.annotation.Nonnull final Precision precision, @javax.annotation.Nonnull final double[] destination, int offset) {
    if (size < offset + (long) destination.length * precision.size) {
      throw new IllegalArgumentException(size + " != " + destination.length * 1l * precision.size);
    }
    if (precision == Precision.Float) {
      @Nonnull float[] data = new float[destination.length];
      read(Precision.Float, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = data[i];
      }
    }
    else {
      CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size), (long) destination.length * precision.size, cudaMemcpyDeviceToHost);
      CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) destination.length * precision.size);
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
  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final float[] destination) {return read(precision, destination, 0);}
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @param offset      the offset
   * @return the cuda ptr
   */
  @javax.annotation.Nonnull
  public CudaMemory read(@javax.annotation.Nonnull final Precision precision, @javax.annotation.Nonnull final float[] destination, int offset) {
    if (size < (long) destination.length * precision.size) {
      throw new IllegalArgumentException(size + " != " + (long) destination.length * precision.size);
    }
    if (precision == Precision.Double) {
      @Nonnull double[] data = new double[destination.length];
      read(Precision.Double, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = (float) data[i];
      }
    }
    else {
      CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size), (long) destination.length * precision.size, cudaMemcpyDeviceToHost);
      CudaMemory.getGpuStats(deviceId).memoryReads.addAndGet((long) destination.length * precision.size);
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
  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final double[] data) {return write(precision, data, 0);}
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @param offset    the offset
   * @return the cuda ptr
   */
  @javax.annotation.Nonnull
  public CudaMemory write(@javax.annotation.Nonnull final Precision precision, @javax.annotation.Nonnull final double[] data, long offset) {
    if (size < ((offset + data.length) * precision.size))
      throw new IllegalArgumentException(String.format("%d != (%d + %d) * %d", size, offset, data.length, precision.size));
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data), (long) data.length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) data.length * precision.size);
    return this;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final float[] data) {return write(precision, data, 0);}
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @param offset    the offset
   * @return the cuda ptr
   */
  @javax.annotation.Nonnull
  public CudaMemory write(@javax.annotation.Nonnull final Precision precision, @javax.annotation.Nonnull final float[] data, long offset) {
    if (size < (offset + data.length) * precision.size)
      throw new IllegalArgumentException(String.format("%d != %d * %d", size, data.length, precision.size));
    CudaSystem.cudaMemcpy(getPtr().withByteOffset(offset * precision.size), precision.getPointer(data), (long) data.length * precision.size, cudaMemcpyKind.cudaMemcpyHostToDevice);
    CudaMemory.getGpuStats(deviceId).memoryWrites.addAndGet((long) data.length * precision.size);
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
  @javax.annotation.Nonnull
  public MemoryType getType() {
    return type;
  }
  
  /**
   * Clear cuda ptr.
   *
   * @return the cuda ptr
   */
  @javax.annotation.Nonnull
  CudaMemory clear() {
    CudaSystem.cudaMemset(getPtr(), 0, size);
    return this;
  }
  
}
