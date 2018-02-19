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

import com.simiacryptus.mindseye.lang.*;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
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
  protected static final Logger logger = LoggerFactory.getLogger(CudaTensorList.class);
  @javax.annotation.Nonnull
  private final int[] dimensions;
  private final int length;
  @javax.annotation.Nonnull
  private final Precision precision;
  @Nullable
  private CudaPtr ptr;
  @Nullable
  private volatile TensorList heapCopy = null;
  
  /**
   * Instantiates a new Cu dnn double tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   */
  private CudaTensorList(@Nullable final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    this.precision = precision;
    if (null == ptr) throw new IllegalArgumentException("ptr");
    if (null == ptr.getPtr()) throw new IllegalArgumentException("ptr.getPtr()");
    this.ptr = ptr;
    this.ptr.addRef();
    this.length = length;
    this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
    assert ptr.size == (long) length * Tensor.dim(dimensions) * precision.size;
    assert ptr.getPtr() != null;
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  }
  
  /**
   * Evict to heap.
   *
   * @param deviceId the device id
   */
  public static void evictToHeap(int deviceId) {
    long size;
    synchronized (CudaTensorList.class) {
      size = RegisteredObjectBase.getLivingInstances(CudaTensorList.class)
        .filter(x -> (x.getDeviceId() == deviceId || deviceId < 0 || x.getDeviceId() < 0))
        .mapToLong(CudaTensorList::evictToHeap).sum();
    }
    logger.info(String.format("Cleared %s bytes from GpuTensorLists for device %s", size, deviceId));
  }
  
  private int getDeviceId() {
    return null == ptr ? -1 : ptr.getDeviceId();
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
  public static CudaTensorList wrap(@javax.annotation.Nonnull final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
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
  public static CudaTensorList create(final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    return new CudaTensorList(ptr, length, dimensions, precision);
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
        if (nativeRight.precision == this.precision) {
          if (nativeRight.heapCopy == null) {
            return gpuAddInPlace(nativeRight);
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
  public synchronized TensorList add(@javax.annotation.Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    assert length() == right.length();
    if (right instanceof ReshapedTensorList) return add(((ReshapedTensorList) right).getInner());
    if (heapCopy == null) {
      if (right instanceof CudaTensorList) {
        @javax.annotation.Nonnull final CudaTensorList nativeRight = (CudaTensorList) right;
        if (nativeRight.precision == this.precision) {
          if (nativeRight.heapCopy == null) {
            return gpuAdd(nativeRight);
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
  
  private CudaTensorList gpuAddInPlace(final CudaTensorList nativeRight) {
    return CudaSystem.eval(gpu -> {
      assert getDimensions().length <= 3;
      int d2 = getDimensions().length < 3 ? 1 : getDimensions()[2];
      int d1 = getDimensions().length < 2 ? 1 : getDimensions()[1];
      int d0 = getDimensions()[0];
      @Nonnull CudaPtr rPtr = nativeRight.getPtr();
      @Nonnull CudaPtr lPtr = CudaTensorList.this.getPtr();
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = CudaSystem.newTensorDescriptor(
        this.precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, getLength(), d2, d1, d0);
      gpu.cudnnAddTensor(
        precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr()
      );
      gpu.registerForCleanup(sizeDescriptor, rPtr, lPtr);
      return CudaTensorList.this;
    });
  }
  
  private CudaTensorList gpuAdd(final CudaTensorList nativeRight) {
    return CudaSystem.eval(gpu -> {
      assert getDimensions().length <= 3;
      int d2 = getDimensions().length < 3 ? 1 : getDimensions()[2];
      int d1 = getDimensions().length < 2 ? 1 : getDimensions()[1];
      int d0 = getDimensions()[0];
      @Nonnull CudaPtr rPtr = nativeRight.getPtr();
      @Nonnull CudaPtr lPtr = CudaTensorList.this.getPtr();
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = CudaSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, this.precision.code);
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = CudaSystem.newTensorDescriptor(
        this.precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, getLength(), d2, d1, d0);
      @Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
      gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
        precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
      gpu.registerForCleanup(opDescriptor, sizeDescriptor, rPtr, lPtr);
      return CudaTensorList.wrap(outputPtr, getLength(), getDimensions(), this.precision);
    });
  }
  
  @Override
  @Nonnull
  public Tensor get(final int i) {
    assertAlive();
    return heapCopy().get(i);
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
   * Gets precision.
   *
   * @return the precision
   */
  @javax.annotation.Nonnull
  public Precision getPrecision() {
    return precision;
  }
  
  /**
   * Inner tensor list.
   *
   * @return the tensor list
   */
  @Nullable
  private TensorList heapCopy() {
    if (null == heapCopy || heapCopy.isFinalized()) {
      synchronized (this) {
        if (null == heapCopy || heapCopy.isFinalized()) {
          final int itemLength = Tensor.dim(getDimensions());
          final Tensor[] output = IntStream.range(0, getLength())
            .mapToObj(dataIndex -> new Tensor(getDimensions()))
            .toArray(i -> new Tensor[i]);
          CudaPtr ptr = getPtr();
          for (int i = 0; i < getLength(); i++) {
            ptr.read(precision, output[i].getData(), i * itemLength);
          }
          ptr.freeRef();
          heapCopy = TensorArray.wrap(output);
        }
      }
    }
    return heapCopy;
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
   * Is native boolean.
   *
   * @return the boolean
   */
  public boolean isNative() {
    return null == heapCopy;
  }
  
  /**
   * Gets heap copy.
   *
   * @return the heap copy
   */
  @Nullable
  public TensorList getHeapCopy() {
    @Nullable TensorList tensorList = heapCopy();
    tensorList.addRef();
    return tensorList;
  }
  
  @Override
  public TensorList copy() {
    return CudaSystem.eval(gpu -> {
      CudaPtr ptr = getPtr();
      CudaPtr copyPtr = ptr.copyTo(gpu.getDeviceNumber());
      ptr.freeRef();
      @javax.annotation.Nonnull CudaTensorList cudaTensorList = new CudaTensorList(copyPtr, getLength(), getDimensions(), precision);
      copyPtr.freeRef();
      return cudaTensorList;
    });
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
  public synchronized long evictToHeap() {
    synchronized (this) {
      if (null == heapCopy()) {
        throw new IllegalStateException();
      }
      if (null != ptr && !ptr.isFinalized() && 1 == ptr.currentRefCount()) {
        long elements = getElements();
        assert 0 < length;
        assert 0 < elements : Arrays.toString(dimensions);
        ptr.freeRef();
        ptr = null;
        return elements * getPrecision().size;
      }
      else {
        return 0;
      }
    }
  }
  
  /**
   * The Ptr.
   *
   * @return the ptr
   */
  @Nonnull
  public CudaPtr getPtr() {
    if ((null == ptr || ptr.isFinalized()) && null != heapCopy && !heapCopy.isFinalized()) {
      synchronized (this) {
        if ((null == ptr || ptr.isFinalized()) && null != heapCopy && !heapCopy.isFinalized()) {
          ptr = CudaPtr.getCudaPtr(precision, heapCopy);
        }
      }
    }
    if (null == ptr) {
      if (null == heapCopy) {
        throw new IllegalStateException("No data");
      }
      else if (heapCopy.isFinalized()) {
        throw new IllegalStateException("Local data has been freed");
      }
    }
    ptr.addRef();
    return ptr;
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
