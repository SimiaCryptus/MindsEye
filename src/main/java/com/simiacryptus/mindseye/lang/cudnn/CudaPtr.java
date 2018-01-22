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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.lang.TimedResult;
import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

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
  
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  /**
   * The Max.
   */
  static final long MAX = Precision.Double.size * (Integer.MAX_VALUE - 1L);
  private static final boolean DEBUG_LIFECYCLE = true;
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
  private final AtomicBoolean isFinalized = new AtomicBoolean(false);
  private volatile StackTraceElement[] finalizedBy = null;
  
  public static final RecycleBin<ManagedPtrParams, CudaPtr> POINTERS = new RecycleBin<ManagedPtrParams, CudaPtr>() {
    @Override
    protected long size(ManagedPtrParams size) {
      return size.size;
    }
    
    @Override
    protected void free(CudaPtr obj) {
      obj.finalize();
    }
    
    @Override
    public CudaPtr create(ManagedPtrParams key) {
      return new CudaPtr(key.size, key.deviceId, key.type);
    }
    
    @Override
    public boolean isProfiling(ManagedPtrParams length) {
      return false;
    }
    
    @Override
    public void reset(CudaPtr data, ManagedPtrParams size) {
    }
  }.setPersistanceMode(PersistanceMode.Weak).setMaxLengthPerBuffer(1e6).setMaxItemsPerBuffer(3).setMinLengthPerBuffer(1);
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   */
  private CudaPtr(final long size, final int deviceId, MemoryType type) {
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
  public static CudaPtr allocate(final int deviceId, final long size, MemoryType type, boolean dirty) {
    CudaPtr obtain = POINTERS.obtain(new ManagedPtrParams(type == MemoryType.Device ? deviceId : -1, size, type));
    if (!dirty) obtain.clear();
    return obtain;
  }
  
  public static void recycle(final CudaPtr obj) {
    POINTERS.recycle(obj, new ManagedPtrParams(obj.deviceId, obj.size, obj.type));
  }
  
  /**
   * Gets cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public static CudaPtr getCudaPtr(final Precision precision, final TensorList data) {
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision() && ((GpuTensorList) data).isNative()) {
      GpuTensorList gpuTensorList = (GpuTensorList) data;
      final CudaPtr ptr = gpuTensorList.ptr.getCudaPtr();
      assert null != ptr;
      assert null != ptr.getPtr() : null == ptr.finalizedBy ? "" : Arrays.stream(ptr.finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get();
      return ptr;
    }
    else {
      final int listLength = data.length();
      final int elementLength = Tensor.dim(data.getDimensions());
      final double[] inputBuffer = RecycleBinLong.DOUBLES.obtain(elementLength * listLength);
      for (int i = 0; i < listLength; i++) {
        final double[] doubles = data.get(i).getData();
        assert elementLength == doubles.length;
        System.arraycopy(doubles, 0, inputBuffer, i * elementLength, elementLength);
      }
      final CudaPtr ptr = CudaPtr.allocate(CuDNN.getDevice(), (long) inputBuffer.length * precision.size, MemoryType.Managed, true).write(precision, inputBuffer);
      RecycleBinLong.DOUBLES.recycle(inputBuffer, inputBuffer.length);
      return ptr;
    }
  }
  
  private static Pointer acquire(int deviceId, long size, MemoryType type, int retries) {
    if (retries < 0) throw new IllegalArgumentException();
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaPtr.MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (deviceId >= 0 && CuDNN.getDevice() != deviceId) throw new IllegalArgumentException();
    final GpuStats metrics = CudaPtr.getGpuStats(deviceId);
    try {
      Pointer pointer = new Pointer();
      type.alloc(size, pointer);
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
  public static Tensor read(final CudaPtr ptr, final Precision precision, final int[] dimensions) {
    CuDNN.cudaDeviceSynchronize();
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
   * Gets current device properties.
   *
   * @return the current device properties
   */
  static cudaDeviceProp getCurrentDeviceProperties() {
    return CuDNN.getDeviceProperties(CuDNN.getDevice());
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
  
  /**
   * Copy to cuda ptr.
   *
   * @param deviceId the device id
   * @return the cuda ptr
   */
  public CudaPtr copyTo(int deviceId) {
    return CuDNN.withDevice(deviceId, () -> {
      CudaPtr copy = allocate(deviceId, size, MemoryType.Managed, false);
      CuDNN.cudaMemcpy(copy.getPtr(), this.getPtr(), size, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
      return copy;
    });
  }
  
  /**
   * Managed managed cuda ptr.
   *
   * @return the managed cuda ptr
   */
  public ManagedCudaPtr managed() {
    return new ManagedCudaPtr(this, PersistanceMode.Strong);
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
  protected void free() {
    if (isActiveObj()) {
      assertAlive();
      if (!isFinalized.getAndSet(true)) {
        if (DEBUG_LIFECYCLE) finalizedBy = Thread.currentThread().getStackTrace();
        getType().free(ptr, deviceId);
        CudaPtr.getGpuStats(deviceId).usedMemory.addAndGet(-size);
      }
    }
  }
  
  /**
   * Assert alive cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr assertAlive() {
    if (null != finalizedBy)
      throw new IllegalStateException(Arrays.stream(finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get());
    if (isFinalized.get()) throw new IllegalStateException();
    return this;
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
      CuDNN.cudaMemcpy(precision.getPointer(destination), getPtr(), size, cudaMemcpyDeviceToHost);
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
      CuDNN.cudaMemcpy(precision.getPointer(destination), getPtr(), size, cudaMemcpyDeviceToHost);
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
  public CudaPtr write(final Precision precision, final double[] data) {
    if (size != (long) data.length * precision.size)
      throw new IllegalArgumentException(String.format("%d != %d * %d", size, data.length, precision.size));
    final Pointer src = precision.getPointer(data);
    CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
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
  public CudaPtr write(final Precision precision, final float[] data) {
    if (size != data.length * precision.size) throw new IllegalArgumentException();
    final Pointer src = precision.getPointer(data);
    CuDNN.cudaMemcpy(getPtr(), src, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
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
  public MemoryType getType() {
    return type;
  }
  
  private CudaPtr clear() {
    CuDNN.cudaMemset(getPtr(), 0, size);
    return this;
  }
  
  private static class ManagedPtrParams {
    public final int deviceId;
    public final long size;
    public final MemoryType type;
    
    private ManagedPtrParams(int deviceId, long size, MemoryType type) {
      this.deviceId = deviceId;
      this.size = size;
      this.type = type;
    }
    
  }

}
