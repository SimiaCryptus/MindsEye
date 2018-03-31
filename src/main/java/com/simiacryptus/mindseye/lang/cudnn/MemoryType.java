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
import com.simiacryptus.mindseye.lang.ReferenceWrapper;
import com.simiacryptus.mindseye.test.TestUtil;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static jcuda.runtime.JCuda.*;

/**
 * The enum Memory type.
 */
public enum MemoryType {
  
  /**
   * The Device.
   */
  Managed {
    public CudaPointer alloc(final long size, final CudaDevice cudaDevice) {
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      CudaPointer pointer = new CudaPointer();
      CudaSystem.handle(CudaSystem.cudaMallocManaged(pointer, size, cudaMemAttachGlobal));
      return pointer;
    }
    
    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }
  
    public MemoryType normalize() {
      return CudaSettings.INSTANCE.isEnableManaged() ? this : Device;
    }
  },
  /**
   * The Device direct.
   */
  Device {
    public CudaPointer alloc(final long size, final CudaDevice cudaDevice) {
      CudaPointer pointer = new CudaPointer();
      CudaSystem.handle(CudaSystem.cudaMalloc(pointer, size));
      return pointer;
    }
    
    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaDevice.cudaFree(deviceId, ptr);
    }
  },
  /**
   * The Host.
   */
  Host {
    public CudaPointer alloc(final long size, final CudaDevice cudaDevice) {
      CudaPointer pointer = new CudaPointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDeviceId());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocDefault));
      }
      else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }
    
    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  },
  /**
   * The Host writeable.
   */
  HostWriteable {
    public CudaPointer alloc(final long size, final CudaDevice cudaDevice) {
      CudaPointer pointer = new CudaPointer();
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDeviceId());
      if (properties.canMapHostMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaHostAlloc(pointer, size, cudaHostAllocWriteCombined));
      }
      else {
        throw new UnsupportedOperationException();
      }
      return pointer;
    }
    
    @Override
    void free(CudaPointer ptr, int deviceId) {
      CudaSystem.cudaFreeHost(ptr);
    }
  };
  
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(MemoryType.class);
  private final Map<Integer, RecycleBin<ReferenceWrapper<CudaPointer>>> cache = new ConcurrentHashMap<>();
  
  /**
   * Free.
   *
   * @param ptr      the ptr
   * @param deviceId the device id
   */
  abstract void free(CudaPointer ptr, int deviceId);
  
  /**
   * Recycle.
   *
   * @param ptr      the ptr
   * @param deviceId the device id
   * @param length   the length
   */
  public void recycle(CudaPointer ptr, int deviceId, final long length) {
    logger.debug(String.format("Recycle %s %s (%s bytes) in device %s via %s", name(), Integer.toHexString(System.identityHashCode(ptr)), length, deviceId, !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller()));
    get(deviceId).recycle(new ReferenceWrapper<>(ptr, x -> {
      logger.debug(String.format("Freed %s %s (%s bytes) in device %s via %s", name(), Integer.toHexString(System.identityHashCode(ptr)), length, deviceId, !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller()));
      CudaMemory.getGpuStats(deviceId).usedMemory.addAndGet(-length);
      MemoryType.this.free(x, deviceId);
    }), length);
  }
  
  /**
   * Get recycle bin.
   *
   * @param device the device
   * @return the recycle bin
   */
  protected RecycleBin<ReferenceWrapper<CudaPointer>> get(int device) {
    return cache.computeIfAbsent(device, d -> {
      logger.info(String.format("Initialize recycle bin %s (device %s)", this, device));
      return new RecycleBin<ReferenceWrapper<CudaPointer>>() {
        @Override
        protected void free(final ReferenceWrapper<CudaPointer> obj) {
          MemoryType.logger.debug(String.format("Freed %s %s in device %s at %s", name(), Integer.toHexString(System.identityHashCode(obj.peek())), device, !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller()));
          obj.destroy();
        }
  
        @Override
        public ReferenceWrapper<CudaPointer> obtain(final long length) {
          assert -1 == device || CudaSystem.getThreadDeviceId() == device;
          ReferenceWrapper<CudaPointer> referenceWrapper = super.obtain(length);
          MemoryType.logger.debug(String.format("Obtained %s %s (%s bytes) in device %s via %s", name(), Integer.toHexString(System.identityHashCode(referenceWrapper.peek())), length, device, !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller()));
          return referenceWrapper;
        }
  
        @Nonnull
        @Override
        public ReferenceWrapper<CudaPointer> create(final long length) {
          assert -1 == device || CudaSystem.getThreadDeviceId() == device;
          CharSequence caller = !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller();
          return CudaDevice.run(gpu -> {
            CudaPointer alloc = MemoryType.this.alloc(length, gpu);
            MemoryType.logger.debug(String.format("Created %s %s (%s bytes) in device %s via %s", name(), Integer.toHexString(System.identityHashCode(alloc)), length, device, caller));
            CudaMemory.getGpuStats(device).usedMemory.addAndGet(length);
            return new ReferenceWrapper<>(alloc, x -> {
              MemoryType.logger.debug(String.format("Freed %s %s (%s bytes) in device %s via %s", name(), Integer.toHexString(System.identityHashCode(alloc)), length, device, !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller()));
              CudaMemory.getGpuStats(device).usedMemory.addAndGet(-length);
              MemoryType.this.free(x, device);
            });
          });
        }
        
        @Override
        public void reset(final ReferenceWrapper<CudaPointer> data, final long size) {
          // There is no need to clean new objects - native memory system doesn't either.
        }
      }.setPersistanceMode(CudaSettings.INSTANCE.memoryCacheMode)
        .setMinLengthPerBuffer(1)
        .setMaxItemsPerBuffer(5)
        .setPurgeFreq(CudaSettings.INSTANCE.getMemoryCacheTTL());
    });
  }
  
  /**
   * Normalize memory type.
   *
   * @return the memory type
   */
  public MemoryType normalize() {
    return this;
  }
  
  /**
   * Purge double.
   *
   * @param device the device
   * @return the double
   */
  public double purge(final int device) {
    double clear = get(device).clear();
    logger.info(String.format("Purged %e bytes from pool for %s (device %s)", clear, this, device));
    return clear;
  }
  
  /**
   * Alloc cached pointer.
   *
   * @param size       the size
   * @param cudaDevice the cuda device
   * @return the pointer
   */
  public CudaPointer allocCached(final long size, final CudaDevice cudaDevice) {
    RecycleBin<ReferenceWrapper<CudaPointer>> recycleBin = get(cudaDevice.deviceId);
    ReferenceWrapper<CudaPointer> wrapper = recycleBin.obtain(size);
    CudaPointer ptr = wrapper.unwrap();
    return ptr;
  }
  
  
  /**
   * Alloc pointer.
   *
   * @param size       the size
   * @param cudaDevice the cuda device
   * @return the pointer
   */
  public abstract CudaPointer alloc(final long size, final CudaDevice cudaDevice);
  
}

