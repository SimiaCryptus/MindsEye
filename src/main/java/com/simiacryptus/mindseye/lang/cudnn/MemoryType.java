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

import static jcuda.runtime.JCuda.*;

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
      cudaDeviceProp properties = CudaPtr.getCurrentDeviceProperties();
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
      cudaDeviceProp properties = CudaPtr.getCurrentDeviceProperties();
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
      cudaDeviceProp properties = CudaPtr.getCurrentDeviceProperties();
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
