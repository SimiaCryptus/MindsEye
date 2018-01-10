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
import com.simiacryptus.mindseye.lang.PersistanceMode;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaMemcpyKind;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.runtime.JCuda.*;

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
  
  private static final boolean lockPci = Boolean.parseBoolean(System.getProperty("lockPci", "true"));
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  private static final long MAX = 4 * GiB;
  private static final Object pciBusLock = new Object();
  private static final boolean useDefaultDir = false;
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
  protected CudaPtr(final long size, final int deviceId, MemoryType type) {this(size, deviceId, type, false);}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param size     the size
   * @param deviceId the device id
   * @param type     the type
   * @param dirty    the dirty
   */
  protected CudaPtr(final long size, final int deviceId, MemoryType type, boolean dirty) {
    super(acquire(deviceId, size, type, dirty));
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
    if (data instanceof GpuTensorList && precision == ((GpuTensorList) data).getPrecision()) {
      final CudaPtr ptr = ((GpuTensorList) data).ptr.getCudaPtr();
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
      final CudaPtr ptr = new CudaPtr(inputBuffer.length * precision.size, deviceId, type, true).write(precision, inputBuffer);
      RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
      return ptr;
    }
  }
  
  private static Pointer acquire(int deviceId, long size, MemoryType type, boolean dirty) {
    if (size < 0) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (size > CudaPtr.MAX) {
      throw new OutOfMemoryError("Allocated block is too large: " + size);
    }
    if (CuDNN.getDevice() != deviceId) throw new IllegalArgumentException();
    final GpuStats metrics = CudaPtr.getGpuStats(deviceId);
    Pointer pointer;
    synchronized (CudaPtr.getPciBusLock()) {
      try {
        pointer = new Pointer();
        type.alloc(size, pointer);
        if (!dirty) {
          CuDNN.handle(CuDNN.cudaMemset(pointer, 0, size));
        }
      } catch (final Exception e) {
        try {
          final long startMemory = metrics.usedMemory.get();
          GpuController.cleanMemory();
          final long freedMemory = startMemory - metrics.usedMemory.get();
          pointer = new Pointer();
          type.alloc(size, pointer);
          if (!dirty) {
            CuDNN.handle(CuDNN.cudaMemset(pointer, 0, size));
          }
          String msg = String.format("Low GPU Memory while allocating %s bytes; %s freed resulting in %s total (triggered by %s)",
                                     size, freedMemory, metrics.usedMemory.get() + size, e.getMessage());
          System.err.println(msg);
        } catch (final Exception e2) {
          cudaDeviceProp properties = getCurrentDeviceProperties();
          String msg = String.format("Error allocating %s bytes; %s currently allocated to device %s; device properties = %s", size, metrics.usedMemory.get(), deviceId, properties);
          throw new com.simiacryptus.mindseye.lang.OutOfMemoryError(msg, e2);
        }
      }
      final long finalMemory = metrics.usedMemory.addAndGet(size);
      metrics.peakMemory.updateAndGet(l -> Math.max(finalMemory, l));
      return pointer;
    }
  }
  
  
  private static cudaDeviceProp getCurrentDeviceProperties() {
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
  public ManagedCudaPtr managed() {return managed(PersistanceMode.Strong);}
  
  /**
   * Managed managed cuda ptr.
   *
   * @param persistanceMode the persistance mode
   * @return the managed cuda ptr
   */
  public ManagedCudaPtr managed(PersistanceMode persistanceMode) {
    return new ManagedCudaPtr(this, persistanceMode);
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
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public CudaPtr read(final Precision precision, final double[] data) {
    synchronized (CudaPtr.getPciBusLock()) {
      if (size != data.length * precision.size) {
        throw new IllegalArgumentException(size + " != " + data.length * precision.size);
      }
      Pointer ptr = getPtr();
      final Pointer dst = precision.getPointer(data);
      int returnCode = CuDNN.cudaMemcpy(dst, ptr, size, useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyDeviceToHost);
//      if(cudaErrorInvalidMemcpyDirection == returnCode) {
//        returnCode = CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyHostToHost);
//      }
      CuDNN.handle(returnCode);
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
      int returnCode = CuDNN.cudaMemcpy(dst, getPtr(), size, useDefaultDir ? cudaMemcpyKind.cudaMemcpyDefault : cudaMemcpyKind.cudaMemcpyDeviceToHost);
//      if(cudaErrorInvalidMemcpyDirection == returnCode) {
//        returnCode = CuDNN.cudaMemcpy(dst, getPtr(), size, cudaMemcpyKind.cudaMemcpyHostToHost);
//      }
      CuDNN.handle(returnCode);
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
  
  /**
   * The enum Memory type.
   */
  public enum MemoryType {
    /**
     * The Device.
     */
    Device {
      @Override
      void alloc(long size, Pointer pointer) {
        if (size < 0) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        if (size > CudaPtr.MAX) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        cudaDeviceProp properties = getCurrentDeviceProperties();
        if (properties.managedMemory == 1) {
          CuDNN.handle(CuDNN.cudaMallocManaged(pointer, size, cudaMemAttachGlobal));
        }
        else {
          CuDNN.handle(CuDNN.cudaMalloc(pointer, size));
        }
      }
      
      @Override
      void free(Pointer ptr, int deviceId) {
        CuDNN.cudaFree(ptr, deviceId);
      }
    },
    /**
     * The Device direct.
     */
    DeviceDirect {
      @Override
      void alloc(long size, Pointer pointer) {
        CuDNN.handle(CuDNN.cudaMalloc(pointer, size));
      }
      
      @Override
      void free(Pointer ptr, int deviceId) {
        CuDNN.cudaFree(ptr, deviceId);
      }
    },
    /**
     * The Host.
     */
    Host {
      @Override
      void alloc(long size, Pointer pointer) {
        if (size < 0) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        if (size > CudaPtr.MAX) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        cudaDeviceProp properties = getCurrentDeviceProperties();
        if (properties.canMapHostMemory == 1) {
          CuDNN.handle(CuDNN.cudaHostAlloc(pointer, size, cudaHostAllocDefault));
        }
        else {
          throw new UnsupportedOperationException();
        }
      }
      
      @Override
      void free(Pointer ptr, int deviceId) {
        CuDNN.cudaFreeHost(ptr);
      }
    },
    /**
     * The Host writeable.
     */
    HostWriteable {
      @Override
      void alloc(long size, Pointer pointer) {
        if (size < 0) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        if (size > CudaPtr.MAX) {
          throw new OutOfMemoryError("Allocated block is too large: " + size);
        }
        cudaDeviceProp properties = getCurrentDeviceProperties();
        if (properties.canMapHostMemory == 1) {
          CuDNN.handle(CuDNN.cudaHostAlloc(pointer, size, cudaHostAllocWriteCombined));
        }
        else {
          throw new UnsupportedOperationException();
        }
      }
      
      @Override
      void free(Pointer ptr, int deviceId) {
        CuDNN.cudaFreeHost(ptr);
      }
    };
    
    /**
     * Alloc.
     *
     * @param size    the size
     * @param pointer the pointer
     */
    abstract void alloc(long size, Pointer pointer);
    
    /**
     * Free.
     *
     * @param ptr      the ptr
     * @param deviceId the device id
     */
    abstract void free(Pointer ptr, int deviceId);
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
