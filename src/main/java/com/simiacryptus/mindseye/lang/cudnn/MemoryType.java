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

import com.simiacryptus.mindseye.lang.RecycleBin;
import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import static jcuda.runtime.JCuda.*;

/**
 * The enum Memory type.
 */
public enum MemoryType {
  
  /**
   * The Device.
   */
  Managed {
    public Pointer alloc(final long size, final CudaDevice cudaDevice) {
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      Pointer pointer = new Pointer();
      CudaSystem.handle(CudaSystem.cudaMallocManaged(pointer, size, cudaMemAttachGlobal));
      return pointer;
    }
    
    @Override
    public void recycle(final Pointer ptr, final int deviceId, final long size) {
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDevice());
      if (properties.managedMemory == 1) {
        super.recycle(ptr, deviceId, size);
      }
      else {
        Device.recycle(ptr, deviceId, size);
      }
    }
    
    @Override
    public Pointer allocCached(final long size, final CudaDevice cudaDevice) {
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDevice());
      if (properties.managedMemory == 1) {
        return super.allocCached(size, cudaDevice);
      }
      else {
        return Device.allocCached(size, cudaDevice);
      }
    }
    
    @Override
    protected RecycleBin<Wrapper<Pointer>> get(final int device) {
      return super.get(-1);
    }
  
    @Override
    void free(Pointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }
  },
  /**
   * The Device direct.
   */
  Device {
    public Pointer alloc(final long size, final CudaDevice cudaDevice) {
      Pointer pointer = new Pointer();
      CudaSystem.handle(CudaSystem.cudaMalloc(pointer, size));
      return pointer;
    }
    
    @Override
    void free(Pointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }
  },
  /**
   * The Host.
   */
  Host {
    public Pointer alloc(final long size, final CudaDevice cudaDevice) {
      Pointer pointer = new Pointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDevice());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocDefault));
      }
      else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }
    
    @Override
    void free(Pointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  },
  /**
   * The Host writeable.
   */
  HostWriteable {
    public Pointer alloc(final long size, final CudaDevice cudaDevice) {
      Pointer pointer = new Pointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDevice());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocWriteCombined));
      }
      else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }
    
    @Override
    void free(Pointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  };
  
  /**
   * Gets memory type.
   *
   * @param deviceId the device id
   * @return the memory type
   */
  @Nonnull
  public static MemoryType getMemoryType(final int deviceId) {
    return -1 == deviceId ? Managed : Device;
  }
  
  /**
   * Free.
   *
   * @param ptr      the ptr
   * @param deviceId the device id
   */
  abstract void free(Pointer ptr, int deviceId);
  
  protected static final Logger logger = LoggerFactory.getLogger(MemoryType.class);
  private final Map<Integer, RecycleBin<Wrapper<Pointer>>> cache = new ConcurrentHashMap<>();
  
  public void recycle(Pointer ptr, int deviceId, final long size) {
    get(deviceId).recycle(new Wrapper<>(ptr, x -> MemoryType.this.free(x, deviceId)), size);
  }
  
  protected RecycleBin<Wrapper<Pointer>> get(int device) {
    return cache.computeIfAbsent(device, d -> {
      logger.info(String.format("Initialize recycle bin %s (device %s)", this, device));
      return new RecycleBin<Wrapper<Pointer>>() {
        @Override
        protected void free(final Wrapper<Pointer> obj) {
          obj.destroy();
        }
        
        @Nonnull
        @Override
        public Wrapper<Pointer> create(final long length) {
          return CudnnHandle.eval(gpu -> new Wrapper<>(MemoryType.this.alloc(length, gpu), x -> MemoryType.this.free(x, device)));
        }
        
        @Override
        public void reset(final Wrapper<Pointer> data, final long size) {
          // There is no need to clean new objects - native memory system doesn't either.
        }
      }.setPersistanceMode(CudaSettings.INSTANCE.memoryCacheMode);
    });
  }
  
  public long purge(final int device) {
    long clear = get(device).clear();
    logger.info(String.format("Purged %s bytes from pool for %s (device %s)", clear, this, device));
    return clear;
  }
  
  public Pointer allocCached(final long size, final CudaDevice cudaDevice) {
    return get(cudaDevice.deviceId).obtain(size).unwrap();
  }
  
  
  public abstract Pointer alloc(final long size, final CudaDevice cudaDevice);
  
  private static class Wrapper<T> {
    final T obj;
    final Consumer<T> destructor;
    final AtomicBoolean isFinalized = new AtomicBoolean(false);
    
    private Wrapper(final T obj, final Consumer<T> destructor) {
      this.obj = obj;
      this.destructor = destructor;
    }
    
    @Override
    protected void finalize() throws Throwable {
      destroy();
      super.finalize();
    }
    
    public void destroy() {
      if (!isFinalized.getAndSet(true)) {
        destructor.accept(obj);
      }
    }
    
    public T unwrap() {
      if (isFinalized.getAndSet(true)) {
        throw new IllegalStateException();
      }
      return obj;
    }
  }
  
}

