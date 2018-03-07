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

import com.simiacryptus.mindseye.lang.RegisteredObjectBase;
import com.simiacryptus.mindseye.lang.ReshapedTensorList;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A TensorList data object stored on a GPU with a configurable precision.
 */
public class CudaTensorList extends RegisteredObjectBase implements TensorList {
  /**
   * The constant logger.
   */
  public static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  
  
  public final StackTraceElement[] createdBy = CudaSettings.INSTANCE.isProfileMemory() ? getStackTrace() : new StackTraceElement[]{};
  
  public static StackTraceElement[] getStackTrace() {
    return java.util.Arrays.stream(Thread.currentThread().getStackTrace())
      .filter(x -> x.getClassName().startsWith("com.simiacryptus.mindseye.") && !x.getClassName().startsWith("com.simiacryptus.mindseye.lang."))
      .limit(1)
      .toArray(i -> new StackTraceElement[i]);
  }
  
  
  
  @javax.annotation.Nonnull
  private final int[] dimensions;
  private final int length;
  /**
   * The Ptr.
   */
  @Nullable
  CudaTensor ptr;
  /**
   * The Heap copy.
   */
  @Nullable
  volatile TensorArray heapCopy = null;
  
  /**
   * Instantiates a new Cu dnn double tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   */
  private CudaTensorList(@Nullable final CudaTensor ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    //assert 1 == ptr.currentRefCount() : ptr.referenceReport(false, false);
    if (null == ptr) throw new IllegalArgumentException("ptr");
    if (null == ptr.memory.getPtr()) throw new IllegalArgumentException("ptr.getPtr()");
    this.ptr = ptr;
    this.ptr.addRef();
    this.length = length;
    this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
    assert ptr.memory.size >= (long) (length - 1) * Tensor.length(dimensions) * precision.size : String.format("%s < %s", ptr.memory.size, (long) length * Tensor.length(dimensions) * precision.size);
    assert ptr.descriptor.batchCount == length;
    assert ptr.descriptor.channels == (dimensions.length < 3 ? 1 : dimensions[2]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.height == (dimensions.length < 2 ? 1 : dimensions[1]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.descriptor.width == (dimensions.length < 1 ? 1 : dimensions[0]) : String.format("%s != (%d,%d,%d,%d)", Arrays.toString(dimensions), ptr.descriptor.batchCount, ptr.descriptor.channels, ptr.descriptor.height, ptr.descriptor.width);
    assert ptr.getPrecision() == precision;
    assert ptr.memory.getPtr() != null;
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }
  
  /**
   * Evict to heap.
   *
   * @param deviceId the device id
   * @return the long
   */
  public static long evictToHeap(int deviceId) {
    long size = RegisteredObjectBase.getLivingInstances(CudaTensorList.class)
      .filter(x -> x.ptr != null && (x.getDeviceId() == deviceId || deviceId < 0 || x.getDeviceId() < 0))
      .mapToLong(CudaTensorList::evictToHeap).sum();
    logger.info(String.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
    return size;
  }
  
  /**
   * Wrap gpu tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   * @return the gpu tensor list
   */
  @javax.annotation.Nonnull
  public static CudaTensorList wrap(@javax.annotation.Nonnull final CudaTensor ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    @javax.annotation.Nonnull CudaTensorList cudaTensorList = new CudaTensorList(ptr, length, dimensions, precision);
    ptr.freeRef();
    return cudaTensorList;
  }
  
  /**
   * Create gpu tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   * @return the gpu tensor list
   */
  public static CudaTensorList create(final CudaTensor ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    return new CudaTensorList(ptr, length, dimensions, precision);
  }
  
  /**
   * Create cuda tensor list.
   *
   * @param ptr        the ptr
   * @param descriptor the descriptor
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   * @return the cuda tensor list
   */
  public static CudaTensorList create(final CudaMemory ptr, CudaDevice.CudaTensorDescriptor descriptor, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    CudaTensor cudaTensor = new CudaTensor(ptr, descriptor, precision);
    CudaTensorList cudaTensorList = new CudaTensorList(cudaTensor, length, dimensions, precision);
    cudaTensor.freeRef();
    return cudaTensorList;
  }
  
  private int getDeviceId() {
    return null == ptr ? -1 : ptr.memory.getDeviceId();
  }
  
  @Override
  public synchronized TensorList addAndFree(@javax.annotation.Nonnull final TensorList right) {
    assertAlive();
    if (right instanceof ReshapedTensorList) return addAndFree(((ReshapedTensorList) right).getInner());
    if (1 < currentRefCount()) {
      TensorList sum = add(right);
      freeRef();
      return sum;
    }
  
    assert length() == right.length();
    if (heapCopy == null) {
      if (right instanceof CudaTensorList) {
        @javax.annotation.Nonnull final CudaTensorList nativeRight = (CudaTensorList) right;
        synchronized (right) {
          if (nativeRight.getPrecision() == this.getPrecision()) {
            if (nativeRight.heapCopy == null) {
              assert (!nativeRight.ptr.equals(CudaTensorList.this.ptr));
              return CudaSystem.eval(gpu -> {
                if (gpu.getDeviceId() == ptr.memory.getDeviceId()) {
                  return gpu.addInPlace(this, nativeRight);
                }
                else {
                  assertAlive();
                  right.assertAlive();
                  TensorList add = add(right);
                  freeRef();
                  return add;
                }
              }, ptr);
            }
          }
        }
      }
    }
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      Tensor a = get(i);
      Tensor b = right.get(i);
      @javax.annotation.Nullable Tensor r = a.addAndFree(b);
      b.freeRef();
      return r;
    }).toArray(i -> new Tensor[i]));
  }
  
  @Override
  public TensorList add(@javax.annotation.Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    assert length() == right.length();
    if (right instanceof ReshapedTensorList) return add(((ReshapedTensorList) right).getInner());
    if (heapCopy == null) {
      if (right instanceof CudaTensorList) {
        @javax.annotation.Nonnull final CudaTensorList nativeRight = (CudaTensorList) right;
        if (nativeRight.getPrecision() == this.getPrecision()) {
          if (nativeRight.heapCopy == null) {
            return CudaSystem.eval(gpu -> {
              return gpu.add(this, nativeRight);
            }, this);
          }
        }
      }
    }
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      Tensor a = get(i);
      Tensor b = right.get(i);
      @javax.annotation.Nullable Tensor r = a.addAndFree(b);
      b.freeRef();
      return r;
    }).toArray(i -> new Tensor[i]));
  }
  
  @Override
  @Nonnull
  public Tensor get(final int i) {
    assertAlive();
    if (heapCopy != null) return heapCopy.get(i);
    return CudaSystem.eval(gpu -> {
      com.simiacryptus.util.lang.TimedResult<com.simiacryptus.mindseye.lang.Tensor> timedResult = com.simiacryptus.util.lang.TimedResult.time(() -> {
        com.simiacryptus.mindseye.lang.Tensor t = new com.simiacryptus.mindseye.lang.Tensor(getDimensions());
        if (this.ptr.isDense()) {
          com.simiacryptus.mindseye.lang.cudnn.CudaMemory memory = this.ptr.getMemory(gpu);
          memory.read(getPrecision(), t.getData(), i * com.simiacryptus.mindseye.lang.Tensor.length(getDimensions()));
          memory.freeRef();
        }
        else {
          this.ptr.read(gpu, i, t, false);
        }
        return t;
      });
      CudaTensorList.logger.debug(String.format("Read %s bytes in %.4f from Tensor %s, GPU at %s, created by %s",
        ptr.size(), timedResult.seconds(), Integer.toHexString(System.identityHashCode(timedResult.result)),
        com.simiacryptus.mindseye.test.TestUtil.toString(CudaTensorList.getStackTrace()).replaceAll("\n", "\n\t"),
        com.simiacryptus.mindseye.test.TestUtil.toString(createdBy).replaceAll("\n", "\n\t")));
      Tensor tensor = timedResult.result;
      return tensor;
    }, CudaTensorList.this);
  }
  
  /**
   * The Dimensions.
   */
  @javax.annotation.Nonnull
  @Override
  public int[] getDimensions() {
    return Arrays.copyOf(dimensions, dimensions.length);
  }
  
  /**
   * The Precision.
   */
  /**
   * Gets precision.
   *
   * @return the precision
   */
  @javax.annotation.Nonnull
  public Precision getPrecision() {
    return null == ptr ? Precision.Double : ptr.getPrecision();
  }
  
  /**
   * Inner tensor list.
   *
   * @return the tensor list
   */
  @Nullable
  private TensorArray heapCopy() {return heapCopy(false);}
  
  /**
   * Inner tensor list.
   *
   * @param avoidAllocations
   * @return the tensor list
   */
  @Nullable
  private TensorArray heapCopy(final boolean avoidAllocations) {
    if (null == heapCopy || heapCopy.isFinalized()) {
      com.simiacryptus.mindseye.lang.TensorArray copy = toHeap(avoidAllocations);
      final com.simiacryptus.mindseye.lang.TensorArray prev;
      synchronized (this) {
        if (null == heapCopy || heapCopy.isFinalized()) {
          prev = this.heapCopy;
          this.heapCopy = copy;
        }
        else {
          prev = null;
        }
      }
      if (null != prev) prev.freeRef();
    }
    return heapCopy;
  }
  
  private TensorArray toHeap(final boolean avoidAllocations) {
    com.simiacryptus.util.lang.TimedResult<com.simiacryptus.mindseye.lang.TensorArray> timedResult = com.simiacryptus.util.lang.TimedResult.time(() -> com.simiacryptus.mindseye.lang.cudnn.CudaDevice.eval(gpu -> {
      if (null == this.ptr) {
        if (null == heapCopy) {
          throw new IllegalStateException("No data");
        }
        else if (heapCopy.isFinalized()) {
          throw new IllegalStateException("Local data has been freed");
        }
      }
      this.ptr.addRef();
      try {
        assert getPrecision() == this.ptr.getPrecision();
        assert getPrecision() == this.ptr.descriptor.dataType;
        final com.simiacryptus.mindseye.lang.Tensor[] output = java.util.stream.IntStream.range(0, getLength())
          .mapToObj(dataIndex -> new com.simiacryptus.mindseye.lang.Tensor(getDimensions()))
          .toArray(i -> new com.simiacryptus.mindseye.lang.Tensor[i]);
        for (int i = 0; i < getLength(); i++) {
          this.ptr.read(gpu, i, output[i], avoidAllocations);
        }
        return com.simiacryptus.mindseye.lang.TensorArray.wrap(output);
      } finally {
        this.ptr.freeRef();
      }
    }, this));
    CudaTensorList.logger.debug(String.format("Read %s bytes in %.4f from Tensor %s on GPU at %s, created by %s",
      ptr.size(), timedResult.seconds(), Integer.toHexString(System.identityHashCode(timedResult.result)),
      com.simiacryptus.mindseye.test.TestUtil.toString(CudaTensorList.getStackTrace()).replaceAll("\n", "\n\t"),
      com.simiacryptus.mindseye.test.TestUtil.toString(createdBy).replaceAll("\n", "\n\t")));
    return timedResult.result;
  }
  
  
  @Override
  public int length() {
    return getLength();
  }
  
  @Override
  public Stream<Tensor> stream() {
    return heapCopy().stream();
  }
  
  /**
   * Gets heap copy.
   *
   * @return the heap copy
   */
  @Nullable
  public TensorArray getHeapCopy() {
    @Nullable TensorArray tensorList = heapCopy();
    tensorList.addRef();
    return tensorList;
  }
  
  @Override
  public TensorList copy() {
    return CudaSystem.eval(gpu -> {
      CudaTensor ptr = gpu.getTensor(this, MemoryType.Managed, false);
      CudaMemory cudaMemory = ptr.getMemory(gpu, MemoryType.Managed);
      CudaMemory copyPtr = cudaMemory.copy(gpu, MemoryType.Managed);
      cudaMemory.freeRef();
      try {
        CudaTensor cudaTensor = new CudaTensor(copyPtr, ptr.descriptor, getPrecision());
        CudaTensorList cudaTensorList = new CudaTensorList(cudaTensor, getLength(), getDimensions(), getPrecision());
        cudaTensor.freeRef();
        return cudaTensorList;
      } finally {
        copyPtr.freeRef();
        ptr.freeRef();
      }
    }, this);
  }
  
  @Override
  protected void _free() {
    synchronized (this) {
      if (null != ptr) {
        ptr.freeRef();
        ptr = null;
      }
      if (null != heapCopy) {
        heapCopy.freeRef();
        heapCopy = null;
      }
    }
  }
  
  /**
   * Evict to heap.
   *
   * @return the long
   */
  public long evictToHeap() {
    if (null == heapCopy(true)) {
      throw new IllegalStateException();
    }
    CudaTensor ptr;
    synchronized (this) {
      ptr = this.ptr;
      this.ptr = null;
    }
    if (null != ptr && !ptr.isFinalized() && 1 == ptr.currentRefCount()) {
      long elements = getElements();
      assert 0 < length;
      assert 0 < elements : Arrays.toString(dimensions);
      ptr.freeRef();
      return elements * getPrecision().size;
    }
    else {
      synchronized (this) {
        if (null != this.ptr) this.ptr.freeRef();
        this.ptr = ptr;
      }
      return 0;
    }
  }
  
  /**
   * The Length.
   *
   * @return the length
   */
  public int getLength() {
    return length;
  }
}
