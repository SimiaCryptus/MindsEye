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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.Pointer;
import jcuda.runtime.cudaMemcpyKind;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A GPU memory segment
 */
public class CudaPtr extends CudaResourceBase<Pointer> {
  
  /**
   * The constant METRICS.
   */
  public static final LoadingCache<Integer, GpuStats> METRICS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, GpuStats>() {
    @Override
    public GpuStats load(final Integer integer) throws Exception {
      return new GpuStats();
    }
  });
  /**
   * The Buffers.
   */
  static final LoadingCache<Integer, RecycleBin<Pointer>> BUFFERS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, RecycleBin<Pointer>>() {
    @Override
    public RecycleBin<Pointer> load(Integer key) throws Exception {
      return new RecycleBin<Pointer>() {
        @Override
        protected void free(Pointer obj) {
          CuDNN.cudaFree(obj);
        }
        
        @Override
        public Pointer create(final long size) {
          Pointer pointer = new Pointer();
          CuDNN.handle(CuDNN.cudaMalloc(pointer, size));
          CuDNN.handle(CuDNN.cudaMemset(pointer, 0, size));
          return pointer;
        }
        
        @Override
        public void reset(final Pointer data, long size) {
          CuDNN.handle(CuDNN.cudaMemset(data, 0, size));
        }
      };
    }
  });
  private static final boolean lockPci = Boolean.parseBoolean(System.getProperty("lockPci", "true"));
  private static final long MAX = 4l * 1024 * 1024 * 1024;
  private static final Object pciBusLock = new Object();
  /**
   * The Size.
   */
  public final long size;
  private final int deviceId;
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   */
  protected CudaPtr(final long size, final int deviceId) {
    super(acquire(deviceId, size));
    this.size = size;
    this.deviceId = deviceId;
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaPtr.MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
  }
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr      the ptr
   * @param size     the size
   * @param deviceId the device id
   */
  protected CudaPtr(final Pointer ptr, final long size, final int deviceId) {
    super(ptr);
    this.size = size;
    this.deviceId = deviceId;
  }
  
  /**
   * Reset.
   */
  public static void reset() {
    BUFFERS.asMap().values().forEach(x -> x.clear());
  }
  
  private static Pointer acquire(int deviceId, long size) {
    final GpuStats metrics = CudaPtr.getGpuStats(deviceId);
    Pointer pointer;
    try {
      pointer = BUFFERS.get(deviceId).obtain(size);
    } catch (final Exception e) {
      try {
        final long startMemory = metrics.usedMemory.get();
        GpuController.cleanMemory();
        final long freedMemory = startMemory - metrics.usedMemory.get();
        pointer = BUFFERS.get(deviceId).obtain(size);
        System.err.println(String.format("Low GPU Memory while allocating %s bytes; %s freed resulting in %s total (triggered by %s)",
                                         size, freedMemory, metrics.usedMemory.get() + size, e.getMessage()));
      } catch (final Exception e2) {
        throw new com.simiacryptus.mindseye.lang.OutOfMemoryError(String.format("Error allocating %s bytes; %s currently allocated to device %s", size, metrics.usedMemory.get(), deviceId), e2);
      }
    }
    final long finalMemory = metrics.usedMemory.addAndGet(size);
    metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
    return pointer;
  }
  
  /**
   * To device as double cuda ptr.
   *
   * @param deviceId  the device id
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public static CudaPtr write(final int deviceId, final Precision precision, final TensorList data) {
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision()) {
      final CudaPtr ptr = ((GpuTensorList) data).ptr;
      assert null != ptr;
      assert null != ptr.getPtr() : null == ptr.finalizedBy ? "" : Arrays.stream(ptr.finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get();
      return ptr;
    }
    else {
      final int listLength = data.length();
      final int elementLength = data.get(0).dim();
      final double[] inputBuffer = RecycleBin.DOUBLES.obtain(elementLength * listLength);
      for (int i = 0; i < listLength; i++) {
        final double[] doubles = data.get(i).getData();
        assert elementLength == doubles.length;
        System.arraycopy(doubles, 0, inputBuffer, i * elementLength, elementLength);
      }
      final CudaPtr ptr = new CudaPtr(inputBuffer.length * precision.size, deviceId).write(precision, inputBuffer);
      RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
      return ptr;
    }
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
    } catch (final ExecutionException e) {
      throw new RuntimeException(e.getCause());
    }
    return devivceMemCtr;
  }
  
  private static Object getPciBusLock() {
    return CudaPtr.lockPci ? CudaPtr.pciBusLock : new Object();
  }
  
  /**
   * From device double tensor.
   *
   * @param ptr        the filter data
   * @param precision  the precision
   * @param dimensions the dimensions  @return the tensor
   * @return the tensor
   */
  public static Tensor read(final CudaPtr ptr, final Precision precision, final int[] dimensions) {
    final Tensor tensor = new Tensor(dimensions);
    switch (precision) {
      case Float:
        final int length = tensor.dim();
        final float[] data = new float[length];
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
  
  @Override
  protected void free() {
    if (isActiveObj()) {
      try {
        BUFFERS.get(deviceId).recycle(ptr, size);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
      CudaPtr.getGpuStats(deviceId).usedMemory.addAndGet(-size);
    }
  }
  
  /**
   * Copy cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr copy() {
    final CudaPtr copy = new CudaPtr(size, deviceId);
    CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), copy.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice));
    return copy;
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr read(final Precision precision, final double[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != data.length * precision.size) {
        throw new IllegalArgumentException(size + " != " + data.length * precision.size);
      }
      final Pointer dst = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToHost));
      CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
      return this;
    }
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr read(final Precision precision, final float[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != data.length * 1l * precision.size) {
        throw new IllegalArgumentException(size + " != " + data.length * 1l * precision.size);
      }
      final Pointer dst = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToHost));
      CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
      return this;
    }
  
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr write(final Precision precision, final double[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != data.length * precision.size) throw new IllegalArgumentException();
      final Pointer src = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice));
      CudaPtr.getGpuStats(deviceId).memoryWrites.addAndGet(size);
      return this;
    }
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr write(final Precision precision, final float[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != data.length * precision.size) throw new IllegalArgumentException();
      final Pointer src = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice));
      CudaPtr.getGpuStats(deviceId).memoryWrites.addAndGet(size);
      return this;
    }
  }
  
  /**
   * The type Gpu stats.
   */
  public static class GpuStats {
    /**
     * The Memory reads.
     */
    public final AtomicLong memoryReads = new AtomicLong(0);
    /**
     * The Memory writes.
     */
    public final AtomicLong memoryWrites = new AtomicLong(0);
    /**
     * The Peak memory.
     */
    public final AtomicLong peakMemory = new AtomicLong(0);
    /**
     * The Used memory.
     */
    public final AtomicLong usedMemory = new AtomicLong(0);
  }
}
