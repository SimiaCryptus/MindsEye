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

import com.simiacryptus.mindseye.lang.RegisteredObjectBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer;
import jcuda.runtime.cudaMemcpyKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

/**
 * A GPU memory segment
 */
public class CudaMemory extends CudaResourceBase<CudaPointer> {
  
  /**
   * The constant METRICS.
   */
  public static final Map<Integer, DeviceMetrics> METRICS = new ConcurrentHashMap<>();
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
  private final MemoryType type;
  private int writtenBy = -1;
  private long writtenAt = System.nanoTime();
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param gpu  the device id
   * @param size the size
   * @param type the type
   */
  CudaMemory(final CudaDevice gpu, final long size, @Nonnull MemoryType type) {this(size, type, gpu.acquire(size, type, 1), gpu.getDeviceId());}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param type     the type
   * @param memory   the memory
   * @param deviceId the device id
   */
  CudaMemory(final long size, @Nonnull MemoryType type, final CudaPointer memory, final int deviceId) {
    super(memory);
    this.size = size;
    this.deviceId = deviceId;
    this.type = type;
  }
  
  /**
   * Clear memory.
   *
   * @param deviceId the device id
   * @return the long
   */
  public static double clearWeakMemory(final int deviceId) {
    logLoad();
    double totalFreed = 0;
    for (final MemoryType type : MemoryType.values()) {
      totalFreed += type.purge(deviceId);
    }
    logLoad();
    return totalFreed;
  }
  
  /**
   * Clear memory double.
   *
   * @param deviceId the device id
   * @return the double
   */
  public static double clearMemory(final int deviceId) {
    double totalFreed = evictMemory(deviceId);
    for (final MemoryType type : MemoryType.values()) {
      totalFreed += type.purge(deviceId);
    }
    if (totalFreed == 0) {
      logger.info(String.format("Nothing Freed - Running Garbage Collector"));
      System.gc();
      totalFreed = evictMemory(0);
    }
    if (totalFreed == 0) {
      logger.info(String.format("Warning: High Active GPU Memory Usage"));
    }
    logLoad();
    return totalFreed;
  }
  
  /**
   * Evict memory long.
   *
   * @param deviceId the device id
   * @return the long
   */
  public static double evictMemory(final int deviceId) {
    logLoad();
    double bytes = RegisteredObjectBase.getLivingInstances(SimpleConvolutionLayer.class).mapToLong(x -> x.evictDeviceData(deviceId)).sum();
    logger.info(String.format("Cleared %e bytes from ConvolutionFilters for device %s", bytes, deviceId));
    double tensorListsFreed = CudaTensorList.evictToHeap(deviceId);
    return tensorListsFreed + bytes;
  }
  
  private static void logLoad() {
    logger.info(String.format("Current Load: %s", METRICS.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> {
      return String.format("%e / %e", (double) e.getValue().activeMemory.get(), (double) e.getValue().usedMemory.get());
    }))));
  }
  
  
  /**
   * Gets gpu stats.
   *
   * @param deviceId the device id
   * @return the gpu stats
   */
  public static DeviceMetrics getGpuStats(final int deviceId) {
    return CudaMemory.METRICS.computeIfAbsent(deviceId, device -> new DeviceMetrics());
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
        final int length = tensor.length();
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
  public CudaMemory copy(CudaDevice deviceId, final MemoryType memoryType) {
    @Nonnull CudaMemory copy = deviceId.allocate(size, memoryType, false);
    CudaSystem.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    return copy;
  }
  
  
  /**
   * Free.
   */
  protected void _free() {
    synchronize();
    if (ptr.getByteOffset() != 0) return;
    CudnnHandle threadHandle = CudaSystem.getThreadHandle();
    if (null != threadHandle) threadHandle.cleanupNative.add(this);
    else release();
  }
  
  @Override
  public void release() {
    if (ptr.getByteOffset() != 0) return;
    if (isActiveObj()) {
      getType().recycle(ptr, deviceId, size);
      ptr = null;
      CudaMemory.getGpuStats(deviceId).activeMemory.addAndGet(-size);
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
  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final double[] destination, int offset) {
    if (size < (long) (offset + destination.length) * precision.size) {
      throw new IllegalArgumentException(String.format("%d < %d + %d", size, (long) destination.length * precision.size, offset));
    }
    if (precision == Precision.Float) {
      @Nonnull float[] data = new float[destination.length];
      read(Precision.Float, data, offset);
      for (int i = 0; i < destination.length; i++) {
        destination[i] = data[i];
      }
    }
    else {
      CudaSystem.run(gpu -> {
        CudaSystem.cudaMemcpy(precision.getPointer(destination), getPtr().withByteOffset((long) offset * precision.size), (long) destination.length * precision.size, cudaMemcpyDeviceToHost);
      });
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
  @Nonnull
  public CudaMemory read(@Nonnull final Precision precision, @Nonnull final float[] destination, int offset) {
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
  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final double[] data, long offset) {
    assert getType() == MemoryType.Managed || getDeviceId() == CudaSystem.getThreadDeviceId();
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
  @Nonnull
  public CudaMemory write(@Nonnull final Precision precision, @Nonnull final float[] data, int offset) {
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
  @Nonnull
  public MemoryType getType() {
    return type;
  }
  
  /**
   * Clear cuda ptr.
   *
   * @return the cuda ptr
   */
  @Nonnull
  CudaMemory clear() {
    CudaSystem.cudaMemset(getPtr(), 0, size);
    return this;
  }
  
  /**
   * With byte offset cuda memory.
   *
   * @param byteOffset the byte offset
   * @return the cuda memory
   */
  public CudaMemory withByteOffset(final int byteOffset) {
    assertAlive();
    final CudaMemory baseMemorySegment = this;
    baseMemorySegment.addRef();
    return new CudaMemory(size - byteOffset, type, ptr.withByteOffset(byteOffset), baseMemorySegment.getDeviceId()) {
      @Override
      protected void _free() {
        baseMemorySegment.freeRef();
      }
      
      @Override
      public void release() {
      }
    };
  }
  
  public CudaMemory dirty(final CudnnHandle gpu) {
    writtenBy = gpu.getDeviceId();
    writtenAt = System.nanoTime();
    return this;
  }
  
  public void synchronize() {
    if (writtenBy >= 0) CudaSystem.synchronize(writtenAt, writtenBy);
  }
  
}
