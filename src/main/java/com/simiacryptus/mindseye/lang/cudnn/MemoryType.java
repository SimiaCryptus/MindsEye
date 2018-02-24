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

import jcuda.Pointer;
import jcuda.runtime.cudaDeviceProp;

import javax.annotation.Nonnull;

import static jcuda.runtime.JCuda.*;

/**
 * The enum Memory type.
 */
public enum MemoryType {
  /**
   * The Device.
   */
  Managed {
    @Override
    void alloc(long size, Pointer pointer, CudaDevice deviceId) {
      if (size < 0) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      if (size > CudaSettings.INSTANCE.getMaxAllocSize()) {
        throw new OutOfMemoryError("Allocated block is too large: " + size);
      }
      cudaDeviceProp properties = CudaDevice.getDeviceProperties(CudaSystem.getThreadDevice());
      if (properties.managedMemory == 1) {
        CudaSystem.handle(CudaSystem.cudaMallocManaged(pointer, size, cudaMemAttachGlobal));
        //CudaSystem.cudaDeviceSynchronize();
      }
      else {
        CudaSystem.handle(CudaSystem.cudaMalloc(pointer, size));
      }
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
    @Override
    void alloc(long size, Pointer pointer, CudaDevice deviceId) {
      CudaSystem.handle(CudaSystem.cudaMalloc(pointer, size));
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
    @Override
    void alloc(long size, Pointer pointer, CudaDevice deviceId) {
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
    @Override
    void alloc(long size, Pointer pointer, CudaDevice deviceId) {
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
   * Alloc.
   *
   * @param size     the size
   * @param pointer  the pointer
   * @param deviceId the device id
   */
  abstract void alloc(long size, Pointer pointer, CudaDevice deviceId);
  
  /**
   * Free.
   *
   * @param ptr      the ptr
   * @param deviceId the device id
   */
  abstract void free(Pointer ptr, int deviceId);
}
