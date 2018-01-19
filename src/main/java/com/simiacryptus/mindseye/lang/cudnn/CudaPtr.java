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
import com.simiacryptus.mindseye.lang.PersistanceMode;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.lang.TimedResult;
import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A GPU memory segment
 */
public class CudaPtr extends CudaResourceBase<Pointer> {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaPtr.class);
  
  /**
   * The constant METRICS.
   */
  public static final LoadingCache<Integer, GpuStats> METRICS = CacheBuilder.newBuilder().build(new CacheLoader<Integer, GpuStats>() {
    @Override
    public GpuStats load(final Integer integer) throws Exception {
      return new GpuStats();
    }
  });
  
  private static final boolean lockPci = Boolean.parseBoolean(System.getProperty("lockPci", "false"));
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  /**
   * The Max.
   */
  static final long MAX = Precision.Double.size * (Integer.MAX_VALUE - 1L);
  private static final Object pciBusLock = new Object();
  private static final boolean useDefaultDir = false;
  /**
   * The constant DISABLE_DIRTY_MEMORY.
   */
  public static boolean DISABLE_DIRTY_MEMORY = false;
  /**
   * The Size.
   */
  public final long size;
  private final int deviceId;
  private final MemoryType type;
  private final boolean dirty;
  private final AtomicBoolean isFinalized = new AtomicBoolean(false);
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   */
  public CudaPtr(final long size, final int deviceId, MemoryType type) {this(size, deviceId, type, false);}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   * @param dirty    the dirty
   */
  protected CudaPtr(final long size, final int deviceId, MemoryType type, boolean dirty) {
    super(acquire(deviceId, size, type, dirty, 1));
    this.size = size;
    this.dirty = dirty;
    this.deviceId = deviceId;
    this.type = type;
  }
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr      the ptr
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   */
  protected CudaPtr(final Pointer ptr, final long size, final int deviceId, MemoryType type) {
    super(ptr);
    this.size = size;
    this.deviceId = deviceId;
    this.dirty = false;
    this.type = type;
  }
  
  /**
   * To device as double cuda ptr.
   *
   * @param deviceId  the device id
   * @param precision the precision
   * @param data      the data
   * @param type      the type
   * @return the cuda ptr
   */
  public static CudaPtr write(final int deviceId, final Precision precision, final TensorList data, MemoryType type) {
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision() && ((GpuTensorList) data).isNative()) {
      GpuTensorList gpuTensorList = (GpuTensorList) data;
      final CudaPtr ptr = gpuTensorList.ptr.getCudaPtr(deviceId);
      assert null != ptr;
      assert null != ptr.getPtr() : null == ptr.finalizedBy ? "" : Arrays.stream(ptr.finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get();
      return ptr;
    }
    else {
      final int listLength = data.length();
      final int elementLength = Tensor.dim(data.getDimensions());
      final double[] inputBuffer = RecycleBin.DOUBLES.obtain(elementLength * listLength);
      for (int i = 0; i < listLength; i++) {
        final double[] doubles = data.get(i).getData();
        assert elementLength == doubles.length;
        System.arraycopy(doubles, 0, inputBuffer, i * elementLength, elementLength);
      }
      final CudaPtr ptr = new CudaPtr((long) inputBuffer.length * precision.size, deviceId, type, true).write(precision, inputBuffer);
      RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
      return ptr;
    }
  }
  
  private static Pointer acquire(int deviceId, long size, MemoryType type, boolean dirty, int retries) {
    if (retries < 0) throw new IllegalArgumentException();
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaPtr.MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (CuDNN.getDevice() != deviceId) throw new IllegalArgumentException();
    final GpuStats metrics = CudaPtr.getGpuStats(deviceId);
    try {
      Pointer pointer = new Pointer();
      synchronized (CudaPtr.getPciBusLock()) {
        type.alloc(size, pointer);
      }
      if (!dirty || DISABLE_DIRTY_MEMORY) {
        CuDNN.handle(CuDNN.cudaMemset(pointer, 0, size));
      }
      final long finalMemory = metrics.usedMemory.addAndGet(size);
      metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      return pointer;
    } catch (final ThreadDeath e) {
      throw e;
    } catch (final Throwable e) {
      if (retries <= 0) throw new RuntimeException(e);
      final long startMemory = metrics.usedMemory.get();
      TimedResult<Void> timedResult = TimedResult.time(() -> {CuDNN.cleanMemory().get();});
      final long freedMemory = startMemory - metrics.usedMemory.get();
      logger.warn(String.format("Low GPU Memory while allocating %s bytes; %s freed in %.4fs resulting in %s total (triggered by %s)",
                                size, freedMemory, timedResult.seconds(), metrics.usedMemory.get(), e.getMessage()));
    }
    if (retries < 0) throw new IllegalStateException();
    return acquire(deviceId, size, type, dirty, retries - 1);
  }
  
  
  /**
   * Gets current device properties.
   *
   * @return the current device properties
   */
  static cudaDeviceProp getCurrentDeviceProperties() {
    return CuDNN.getDeviceProperties(CuDNN.getDevice());
  }
  
  /**
   * To device as double cuda ptr.
   *
   * @param deviceId  the device id
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public static CudaPtr write(final int deviceId, final Precision precision, final TensorList data) {return write(deviceId, precision, data, MemoryType.Device);}
  
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
  
  /**
   * Managed managed cuda ptr.
   *
   * @return the managed cuda ptr
   */
  public ManagedCudaPtr managed() {
    return new ManagedCudaPtr(this, PersistanceMode.Strong);
  }
  
  @Override
  protected void free() {
    if (isActiveObj() && !isFinalized.getAndSet(true)) {
      synchronized (CudaPtr.getPciBusLock()) {
        getType().free(ptr, deviceId);
        CudaPtr.getGpuStats(deviceId).usedMemory.addAndGet(-size);
      }
    }
  }
  
  /**
   * Copy cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr copy() {
    synchronized (CudaPtr.getPciBusLock()) {
      final CudaPtr copy = new CudaPtr(size, deviceId, getType());
      int returnCode = CuDNN.cudaMemcpy(getPtr(), copy.getPtr(), size, useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyDeviceToDevice);
//      if(cudaErrorInvalidMemcpyDirection == returnCode) {
//        returnCode = CuDNN.cudaMemcpy(getPtr(), copy.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
//      }
      CuDNN.handle(returnCode);
      return copy;
    }
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision   the precision
   * @param destination the data
   * @return the cuda ptr
   */
  public CudaPtr read(final Precision precision, final double[] destination) {
    if (size != destination.length * 1l * precision.size) {
      throw new IllegalArgumentException(size + " != " + destination.length * 1l * precision.size);
    }
    if (precision == Precision.Float) {
      float[] data = new float[destination.length];
      read(Precision.Float, data);
      for (int i = 0; i < data.length; i++) {
        destination[i] = data[i];
      }
    }
    else {
      synchronized (CudaPtr.getPciBusLock()) {
       int returnCode = CuDNN.cudaMemcpy(precision.getPointer(destination), getPtr(),
                                          size,
                                          useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyDeviceToHost);
        //      if(cudaErrorInvalidMemcpyDirection == returnCode) {
        //        returnCode = CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyHostToHost);
        //      }
        CuDNN.handle(returnCode);
        CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
      }
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
  public CudaPtr read(final Precision precision, final float[] destination) {
    if (size != destination.length * 1l * precision.size) {
      throw new IllegalArgumentException(size + " != " + destination.length * 1l * precision.size);
    }
    if (precision == Precision.Double) {
      double[] data = new double[destination.length];
      read(Precision.Double, data);
      for (int i = 0; i < data.length; i++) {
        destination[i] = (float) data[i];
      }
    }
    else {
      synchronized (CudaPtr.getPciBusLock()) {
       int returnCode = CuDNN.cudaMemcpy(precision.getPointer(destination), getPtr(), 
                size, 
                useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyDeviceToHost);
        //      if(cudaErrorInvalidMemcpyDirection == returnCode) {
        //        returnCode = CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyHostToHost);
        //      }
        CuDNN.handle(returnCode);
        CudaPtr.getGpuStats(deviceId).memoryReads.addAndGet(size);
      }
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
  public CudaPtr write(final Precision precision, final double[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != (long) data.length * precision.size)
        throw new IllegalArgumentException(String.format("%d != %d * %d", size, data.length, precision.size));
      final Pointer src = precision.getPointer(data);
      int returnCode = CuDNN.cudaMemcpy(getPtr(), src, size, useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyHostToDevice);
//      if(cudaErrorInvalidMemcpyDirection == returnCode) {
//        returnCode = CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
//      }
      CuDNN.handle(returnCode);
      
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
      int returnCode = CuDNN.cudaMemcpy(getPtr(), src, size, useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyHostToDevice);
//      if(cudaErrorInvalidMemcpyDirection == returnCode) {
//        returnCode = CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
//      }
      CuDNN.handle(returnCode);
      CudaPtr.getGpuStats(deviceId).memoryWrites.addAndGet(size);
      return this;
    }
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
  public MemoryType getType() {
    return type;
  }
  
  /**
   * Is dirty boolean.
   *
   * @return the boolean
   */
  public boolean isDirty() {
    return dirty;
  }
  
}
