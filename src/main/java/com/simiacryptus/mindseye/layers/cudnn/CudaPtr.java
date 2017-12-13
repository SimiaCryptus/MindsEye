/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.DoubleArrays;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.Pointer;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.runtime.cudaMemcpyKind.*;

/**
 * The type Cuda ptr.
 */
public class CudaPtr extends CudaResource<Pointer> {
  
  /**
   * The constant METRICS.
   */
  public static final LoadingCache<Integer, GpuStats> METRICS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, GpuStats>() {
    @Override
    public GpuStats load(Integer integer) throws Exception {
      return new GpuStats();
    }
  });
  private static final boolean lockPci = Boolean.parseBoolean(System.getProperty("lockPci", "true"));
  private static final long MAX = 4l * 1024 * 1024 * 1024;
  private static Object pciBusLock = new Object();
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
  protected CudaPtr(long size, int deviceId) {
    super(new Pointer(), CuDNN::cudaFree);
    this.size = size;
    this.deviceId = deviceId;
    GpuStats metrics = getGpuStats(deviceId);
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    try {
      CuDNN.handle(CuDNN.cudaMalloc(this.getPtr(), size));
    } catch (Exception e) {
      try {
        long startMemory = metrics.usedMemory.get();
        System.gc(); // Force any dead objects to be finalized
        System.runFinalization();
        long freedMemory = startMemory - metrics.usedMemory.get();
        CuDNN.handle(CuDNN.cudaMalloc(this.getPtr(), size));
        System.err.println(String.format("Low GPU Memory while allocating %s bytes; %s freed resulting in %s total (triggered by %s)",
          size, freedMemory, metrics.usedMemory.get() + size, e.getMessage()));
      } catch (Exception e2) {
        throw new com.simiacryptus.mindseye.lang.OutOfMemoryError(String.format("Error allocating %s bytes; %s currently allocated to device %s", size, metrics.usedMemory.get(), deviceId), e2);
      }
    }
    long finalMemory = metrics.usedMemory.addAndGet(size);
    metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
    CuDNN.handle(CuDNN.cudaMemset(this.getPtr(), 0, size));
  }
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr      the ptr
   * @param size     the size
   * @param deviceId the device id
   */
  protected CudaPtr(Pointer ptr, long size, int deviceId) {
    super(ptr, x -> 0);
    this.size = size;
    this.deviceId = deviceId;
  }
  
  private static Object getPciBusLock() {
    return lockPci ? pciBusLock : new Object();
  }
  
  private static void setPciBusLock(Object pciBusLock) {
    CudaPtr.pciBusLock = pciBusLock;
  }
  
  /**
   * Gets gpu stats.
   *
   * @param deviceId the device id
   * @return the gpu stats
   */
  public static GpuStats getGpuStats(int deviceId) {
    GpuStats devivceMemCtr;
    try {
      devivceMemCtr = METRICS.get(deviceId);
    } catch (ExecutionException e) {
      throw new RuntimeException(e.getCause());
    }
    return devivceMemCtr;
  }
  
  /**
   * From device double tensor.
   *
   * @param ptr        the filter data
   * @param precision  the precision
   * @param dimensions the dimensions  @return the tensor
   * @return the tensor
   */
  public static Tensor read(CudaPtr ptr, Precision precision, int[] dimensions) {
    final Tensor tensor = new Tensor(dimensions);
    switch (precision) {
      case Float:
        int length = tensor.dim();
        float[] data = new float[length];
        ptr.read(precision, data);
        double[] doubles = tensor.getData();
        for (int i = 0; i < length; i++) doubles[i] = data[i];
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
   * To device as double cuda ptr.
   *
   * @param deviceId  the device id
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public static CudaPtr write(int deviceId, Precision precision, TensorList data) {
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision()) {
      CudaPtr ptr = ((GpuTensorList) data).ptr;
      assert null != ptr;
      assert null != ptr.getPtr() : null == ptr.finalizedBy ? "" : Arrays.stream(ptr.finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get();
      return ptr;
    }
    else {
      int listLength = data.length();
      int elementLength = data.get(0).dim();
      final double[] inputBuffer = DoubleArrays.obtain(elementLength * listLength);
      for (int i = 0; i < listLength; i++) {
        double[] doubles = data.get(i).getData();
        assert elementLength == doubles.length;
        System.arraycopy(doubles, 0, inputBuffer, i * elementLength, elementLength);
      }
      CudaPtr ptr = new CudaPtr(inputBuffer.length * precision.size, deviceId).write(precision, inputBuffer);
      DoubleArrays.recycle(inputBuffer);
      return ptr;
    }
  }
  
  @Override
  protected void free() {
    if (isActiveObj()) {
      super.free();
      getGpuStats(deviceId).usedMemory.addAndGet(-size);
    }
  }
  
  /**
   * Copy cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr copy() {
    CudaPtr copy = new CudaPtr(size, deviceId);
    CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), copy.getPtr(), size, cudaMemcpyDeviceToDevice));
    return copy;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr write(Precision precision, float[] data) {
    synchronized (getPciBusLock()) {
      if (this.size != data.length * precision.size) throw new IllegalArgumentException();
      Pointer src = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyHostToDevice));
      getGpuStats(deviceId).memoryWrites.addAndGet(size);
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
  public CudaPtr write(Precision precision, double[] data) {
    synchronized (getPciBusLock()) {
      if (this.size != data.length * precision.size) throw new IllegalArgumentException();
      Pointer src = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyHostToDevice));
      getGpuStats(deviceId).memoryWrites.addAndGet(size);
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
  public CudaPtr read(Precision precision, double[] data) {
    synchronized (getPciBusLock()) {
      if (this.size != data.length * precision.size) {
        throw new IllegalArgumentException(this.size + " != " + data.length * precision.size);
      }
      Pointer dst = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyDeviceToHost));
      getGpuStats(deviceId).memoryReads.addAndGet(size);
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
  public CudaPtr read(Precision precision, float[] data) {
    synchronized (getPciBusLock()) {
      if (this.size != data.length * 1l * precision.size) {
        throw new IllegalArgumentException(this.size + " != " + (data.length * 1l * precision.size));
      }
      Pointer dst = precision.getPointer(data);
      CuDNN.handle(CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyDeviceToHost));
      getGpuStats(deviceId).memoryReads.addAndGet(size);
      return this;
    }
    
  }
  
  /**
   * The type Gpu stats.
   */
  public static class GpuStats {
    /**
     * The Used memory.
     */
    public final AtomicLong usedMemory = new AtomicLong(0);
    /**
     * The Peak memory.
     */
    public final AtomicLong peakMemory = new AtomicLong(0);
    /**
     * The Memory writes.
     */
    public final AtomicLong memoryWrites = new AtomicLong(0);
    /**
     * The Memory reads.
     */
    public final AtomicLong memoryReads = new AtomicLong(0);
  }
}
