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
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.lang.TimedResult;

import javax.annotation.Nonnull;
import java.util.function.Function;

/**
 * The type Cuda tensor.
 */
public class CudaTensor extends ReferenceCountingBase implements CudaSystem.CudaDeviceResource {
  
  /**
   * The Descriptor.
   */
  public final CudaDevice.CudaTensorDescriptor descriptor;
  /**
   * The Created by.
   */
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE.isProfileMemoryIO() ? CudaTensorList.getStackTrace() : new StackTraceElement[]{};
  /**
   * The Memory.
   */
  final CudaMemory memory;
  
  /**
   * Instantiates a new Cuda tensor.
   *
   * @param memory     the memory
   * @param descriptor the descriptor
   * @param precision  the precision
   */
  public CudaTensor(final CudaMemory memory, final CudaDevice.CudaTensorDescriptor descriptor, final Precision precision) {
    this.memory = memory;
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
   * Gets memory.
   *
   * @param cudaDevice the cuda device
   * @return the memory
   */
  public CudaMemory getMemory(final CudaDevice cudaDevice) {
    return getMemory(cudaDevice, MemoryType.Device);
  }
  
  /**
   * Gets memory.
   *
   * @param cudaDevice the cuda device
   * @param memoryType the memory type
   * @return the memory
   */
  public CudaMemory getMemory(final CudaDevice cudaDevice, final MemoryType memoryType) {
    if (memory.getType() == MemoryType.Managed) {
      memory.addRef();
      return memory;
    }
    else if (cudaDevice.getDeviceId() == memory.getDeviceId()) {
      memory.addRef();
      return memory;
    }
    else {
      TimedResult<CudaMemory> timedResult = TimedResult.time(() -> memory.copy(cudaDevice, memoryType));
      CudaTensorList.logger.debug(String.format("Copy %s bytes in %.4f from Tensor %s on GPU %s to %s at %s, created by %s",
        memory.size, timedResult.seconds(), Integer.toHexString(System.identityHashCode(this)), memory.getDeviceId(), cudaDevice.getDeviceId(),
        TestUtil.toString(CudaTensorList.getStackTrace()).replaceAll("\n", "\n\t"),
        TestUtil.toString(createdBy).replaceAll("\n", "\n\t")));
      return timedResult.result;
    }
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
    if (isDense()) {
      addRef();
      return this;
    }
    TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
      CudaDevice.CudaTensorDescriptor destDescriptor = gpu.newTensorDescriptor(
        getPrecision(), this.descriptor.batchCount, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
        this.descriptor.channels * this.descriptor.height * this.descriptor.width, this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
      CudaMemory destMemory = gpu.allocate(destDescriptor.nStride * destDescriptor.batchCount * getPrecision().size, getType(), true);
      CudaMemory memory = getMemory(gpu);
      gpu.cudnnTransformTensor(
        getPrecision().getPointer(1.0), this.descriptor.getPtr(), memory.getPtr(),
        getPrecision().getPointer(0.0), destDescriptor.getPtr(), destMemory.getPtr());
      destMemory.dirty(gpu);
      memory.freeRef();
      return CudaTensor.wrap(destMemory, destDescriptor, getPrecision());
    });
    CudaTensor cudaTensor = timedResult.result;
    assert cudaTensor.isDense();
    CudaTensorList.logger.debug(String.format("Densified %s bytes in %.4f from GPU %s at %s, created by %s",
      cudaTensor.memory.size, timedResult.seconds(), Integer.toHexString(System.identityHashCode(this)),
      TestUtil.toString(CudaTensorList.getStackTrace()).replaceAll("\n", "\n\t"),
      TestUtil.toString(createdBy).replaceAll("\n", "\n\t")));
    return cudaTensor;
  
  }
  
  /**
   * Read.
   *
   * @param gpu              the gpu
   * @param index            the index
   * @param result           the result
   * @param avoidAllocations the avoid allocations
   */
  @Nonnull
  public void read(final CudnnHandle gpu, final int index, final Tensor result, final boolean avoidAllocations) {
    if (isDense()) {
      CudaSystem.withDevice(memory.getDeviceId(), dev -> {
        CudaMemory memory = getMemory(dev);
        memory.read(descriptor.dataType, result.getData(), index * descriptor.nStride);
        memory.freeRef();
      });
    }
    else if (avoidAllocations) {
      int size = (descriptor.channels - 1) * descriptor.cStride +
        (descriptor.height - 1) * descriptor.hStride +
        (descriptor.width - 1) * descriptor.wStride + 1;
      double[] buffer = RecycleBin.DOUBLES.obtain(size);
      try {
        memory.read(descriptor.dataType, buffer, descriptor.nStride * index);
        result.setByCoord(c -> {
          int[] coords = c.getCoords();
          int x = coords.length < 1 ? 1 : coords[0];
          int y = coords.length < 2 ? 1 : coords[1];
          int z = coords.length < 3 ? 1 : coords[2];
          return buffer[x * descriptor.wStride + y * descriptor.hStride + z * descriptor.cStride];
        });
      } finally {
        RecycleBin.DOUBLES.recycle(buffer, buffer.length);
      }
    }
    else {
      withDense(gpu, index, mem -> mem.read(this.descriptor.dataType, result.getData()));
    }
  }
  
  /**
   * With dense t.
   *
   * @param <T>    the type parameter
   * @param gpu    the gpu
   * @param index  the index
   * @param result the result
   * @return the t
   */
  @Nonnull
  public <T> T withDense(final CudnnHandle gpu, final int index, final Function<CudaMemory, T> result) {
    int deviceId = memory.getDeviceId();
    Function<CudaDevice, T> fn = dev -> {
      assertAlive();
      assert this.descriptor.dataType == getPrecision();
      CudaDevice.CudaTensorDescriptor sourceDescriptor = dev.newTensorDescriptor(
        this.descriptor.dataType, 1, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
        this.descriptor.nStride, this.descriptor.cStride, this.descriptor.hStride, this.descriptor.wStride);
      CudaDevice.CudaTensorDescriptor destDescriptor = dev.newTensorDescriptor(
        this.descriptor.dataType, 1, this.descriptor.channels, this.descriptor.height, this.descriptor.width,
        this.descriptor.channels * this.descriptor.height * this.descriptor.width, this.descriptor.height * this.descriptor.width, this.descriptor.width, 1);
      try {
        CudaMemory cudaMemory = dev.allocate(destDescriptor.nStride * this.descriptor.dataType.size, MemoryType.Device, true);
        CudaMemory memory = getMemory(dev);
        try {
          gpu.cudnnTransformTensor(
            this.descriptor.dataType.getPointer(1.0), sourceDescriptor.getPtr(), memory.getPtr().withByteOffset(index * this.descriptor.nStride * getPrecision().size),
            this.descriptor.dataType.getPointer(0.0), destDescriptor.getPtr(), cudaMemory.getPtr());
          cudaMemory.dirty(gpu);
          return result.apply(cudaMemory);
        } finally {
          memory.freeRef();
          cudaMemory.freeRef();
        }
      } finally {
        sourceDescriptor.freeRef();
        destDescriptor.freeRef();
      }
    };
    if (0 > deviceId || gpu.getDeviceId() == deviceId) return fn.apply(gpu);
    else return CudaSystem.withDevice(deviceId, fn);
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
   *
   * @return the type
   */
  public MemoryType getType() {
    return memory.getType();
  }
  
  /**
   * The Descriptor.
   *
   * @return the device id
   */
  public int getDeviceId() {
    return memory.getDeviceId();
  }
  
  /**
   * The Precision.
   *
   * @return the precision
   */
  public Precision getPrecision() {
    return descriptor.dataType;
  }
  
  /**
   * Copy and free cuda tensor.
   *
   * @param device the device
   * @param type   the type
   * @return the cuda tensor
   */
  public CudaTensor copyAndFree(final CudaDevice device, final MemoryType type) {
    return new CudaTensor(memory.copy(device, type), descriptor, getPrecision());
  }
  
  /**
   * Size long.
   *
   * @return the long
   */
  public long size() {
    return memory.size;
  }
}
