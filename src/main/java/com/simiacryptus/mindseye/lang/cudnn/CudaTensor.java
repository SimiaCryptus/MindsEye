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
import com.simiacryptus.mindseye.lang.Tensor;

import javax.annotation.Nonnull;
import java.util.function.Function;

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
   * The Precision.
   */
  public final Precision precision;
  
  /**
   * Instantiates a new Cuda tensor.
   *
   * @param memory     the memory
   * @param descriptor the descriptor
   * @param precision  the precision
   */
  public CudaTensor(final CudaMemory memory, final CudaDevice.CudaTensorDescriptor descriptor, final Precision precision) {
    this.memory = memory;
    this.precision = precision;
    this.memory.addRef();
    this.descriptor = descriptor;
    this.descriptor.addRef();
    assert memory.size >= (long) precision.size * descriptor.nStride * (descriptor.batchCount - 1) : String.format("%d != %d", memory.size, (long) precision.size * descriptor.nStride * descriptor.batchCount);
    assert this.descriptor.dataType == precision;
  }
  
  /**
   * Wrap cuda tensor.
   *
   * @param ptr        the ptr
   * @param descriptor the descriptor
   * @param precision  the precision
   * @return the cuda tensor
   */
  public static CudaTensor wrap(final CudaMemory ptr, final CudaDevice.CudaTensorDescriptor descriptor, final Precision precision) {
    CudaTensor cudaTensor = new CudaTensor(ptr, descriptor, precision);
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
  
  /**
   * Move to cuda tensor.
   *
   * @param cudaDevice the cuda device
   * @param memoryType the memory type
   * @return the cuda tensor
   */
  public CudaTensor moveTo(final CudaDevice cudaDevice, final MemoryType memoryType) {
    descriptor.addRef();
    memory.addRef();
    CudaTensor wrap = CudaTensor.wrap(memory.moveTo(cudaDevice, memoryType), descriptor, precision);
    freeRef();
    return wrap;
  }
  
  /**
   * Gets dense and free.
   *
   * @param gpu the gpu
   * @return the dense and free
   */
  public CudaTensor getDenseAndFree(CudnnHandle gpu) {
    CudaTensor result;
    if (isDense()) {
      result = this;
    }
    else {
      result = getDense(gpu);
      freeRef();
    }
    return result;
  }
  
  /**
   * Gets dense.
   *
   * @param gpu the gpu
   * @return the dense
   */
  public CudaTensor getDense(CudnnHandle gpu) {
    assertAlive();
    assert gpu.getDeviceId() == getDeviceId() || getType() == MemoryType.Managed;
    if (isDense()) {
      addRef();
      return this;
    }
    CudaDevice.CudaTensorDescriptor destDescriptor = gpu.newTensorDescriptor(
      precision, this.descriptor.batchCount, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
      this.descriptor.channels * this.descriptor.height * this.descriptor.width, this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
    CudaMemory destMemory = gpu.allocate(destDescriptor.nStride * destDescriptor.batchCount * precision.size, getType(), true);
    gpu.cudnnTransformTensor(
      precision.getPointer(1.0), this.descriptor.getPtr(), memory.getPtr(),
      precision.getPointer(0.0), destDescriptor.getPtr(), destMemory.getPtr());
    CudaTensor cudaTensor = CudaTensor.wrap(destMemory, destDescriptor, precision);
    assert cudaTensor.isDense();
    return cudaTensor;
  
  }
  
  @Nonnull
  public void read(final CudnnHandle gpu, final int index, final Tensor result) {
    if (isDense()) {
      memory.read(descriptor.dataType, result.getData(), index * descriptor.nStride);
    }
    else {
      withDense(gpu, index, mem -> mem.read(this.descriptor.dataType, result.getData()));
    }
  }
  
  @Nonnull
  public <T> T withDense(final CudnnHandle gpu, final int index, final Function<CudaMemory, T> result) {
    assertAlive();
    assert this.descriptor.dataType == precision;
    assert gpu.getDeviceId() == getDeviceId() || getType() == MemoryType.Managed;
    CudaDevice.CudaTensorDescriptor sourceDescriptor = gpu.newTensorDescriptor(
      this.descriptor.dataType, 1, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
      this.descriptor.nStride, this.descriptor.cStride, this.descriptor.hStride, this.descriptor.wStride);
    CudaDevice.CudaTensorDescriptor destDescriptor = gpu.newTensorDescriptor(
      this.descriptor.dataType, 1, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
      this.descriptor.channels * this.descriptor.height * this.descriptor.width, this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
    try {
      CudaMemory cudaMemory = gpu.allocate(destDescriptor.nStride * this.descriptor.dataType.size, MemoryType.Device, true);
      gpu.cudnnTransformTensor(
        this.descriptor.dataType.getPointer(1.0), sourceDescriptor.getPtr(), memory.getPtr().withByteOffset(index * this.descriptor.nStride * precision.size),
        this.descriptor.dataType.getPointer(0.0), destDescriptor.getPtr(), cudaMemory.getPtr());
      try {
        return result.apply(cudaMemory);
      } finally {
        cudaMemory.freeRef();
      }
    } finally {
      sourceDescriptor.freeRef();
      destDescriptor.freeRef();
    }
  }
  
  /**
   * Is dense boolean.
   *
   * @return the boolean
   */
  public boolean isDense() {
    if (descriptor.nStride != descriptor.channels * descriptor.height * descriptor.width) return false;
    if (descriptor.cStride != descriptor.height * descriptor.width) return false;
    if (descriptor.hStride != descriptor.width) return false;
    return descriptor.wStride == 1;
  }
  
  /**
   * The Descriptor.
   */
  public MemoryType getType() {
    return memory.getType();
  }
  
  /**
   * The Descriptor.
   */
  public int getDeviceId() {
    return memory.getDeviceId();
  }
}
