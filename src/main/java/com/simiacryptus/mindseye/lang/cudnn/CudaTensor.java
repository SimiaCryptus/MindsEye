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

import com.simiacryptus.mindseye.lang.ReferenceCountingBase;

/**
 * The type Cuda tensor.
 */
public class CudaTensor extends ReferenceCountingBase {
  /**
   * The Memory.
   */
  public final CudaMemory memory;
  /**
   * The Descriptor.
   */
  public final CudaDevice.CudaTensorDescriptor descriptor;
  
  /**
   * Instantiates a new Cuda tensor.
   *
   * @param memory     the memory
   * @param descriptor the descriptor
   */
  public CudaTensor(final CudaMemory memory, final CudaDevice.CudaTensorDescriptor descriptor) {
    this.memory = memory;
    this.memory.addRef();
    this.descriptor = descriptor;
    this.descriptor.addRef();
  }
  
  /**
   * Wrap cuda tensor.
   *
   * @param ptr        the ptr
   * @param descriptor the descriptor
   * @return the cuda tensor
   */
  public static CudaTensor wrap(final CudaMemory ptr, final CudaDevice.CudaTensorDescriptor descriptor) {
    CudaTensor cudaTensor = new CudaTensor(ptr, descriptor);
    ptr.freeRef();
    descriptor.freeRef();
    return cudaTensor;
  }
  
  @Override
  protected void _free() {
    memory.freeRef();
    descriptor.freeRef();
    super._free();
  }
  
  public CudaTensor moveTo(final CudaDevice cudaDevice, final MemoryType memoryType) {
    descriptor.addRef();
    memory.addRef();
    CudaTensor wrap = CudaTensor.wrap(memory.moveTo(cudaDevice, memoryType), descriptor);
    freeRef();
    return wrap;
  }
}
