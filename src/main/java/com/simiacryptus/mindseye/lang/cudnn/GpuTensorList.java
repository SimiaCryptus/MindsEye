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
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A TensorList data object stored on a GPU with a configurable precision.
 */
public class GpuTensorList extends ReferenceCountingBase implements TensorList {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(GpuTensorList.class);
  private static final ConcurrentLinkedDeque<WeakReference<GpuTensorList>> INSTANCES = new ConcurrentLinkedDeque<>();
  private static long lastCleanTime = 0;
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
  private GpuTensorList(@Nullable final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
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
    @javax.annotation.Nonnull GpuTensorList self = this;
    addInstance(self);
  }
  
  /**
   * Evict all to heap.
   */
  public static void evictAllToHeap() {
    synchronized (INSTANCES) {
      @javax.annotation.Nonnull Iterator<WeakReference<GpuTensorList>> iterator = INSTANCES.iterator();
      while (iterator.hasNext()) {
        @Nullable GpuTensorList next = iterator.next().get();
        if (null != next && !next.isFinalized()) next.evictToHeap();
      }
    }
  }
  
  private static void addInstance(GpuTensorList self) {
    long now = System.currentTimeMillis();
    if (now - lastCleanTime > 1000) {
      synchronized (INSTANCES) {
        @javax.annotation.Nonnull Iterator<WeakReference<GpuTensorList>> iterator = INSTANCES.iterator();
        while (iterator.hasNext()) {
          @Nullable GpuTensorList next = iterator.next().get();
          if (null == next) iterator.remove();
        }
        lastCleanTime = now;
      }
    }
    INSTANCES.add(new WeakReference<>(self));
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
  public static GpuTensorList wrap(@javax.annotation.Nonnull final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    @javax.annotation.Nonnull GpuTensorList gpuTensorList = new GpuTensorList(ptr, length, dimensions, precision);
    ptr.freeRef();
    return gpuTensorList;
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
  public static GpuTensorList create(final CudaPtr ptr, final int length, @javax.annotation.Nonnull final int[] dimensions, @javax.annotation.Nonnull final Precision precision) {
    return new GpuTensorList(ptr, length, dimensions, precision);
  }
  
  @Override
  public synchronized TensorList addAndFree(@javax.annotation.Nonnull final TensorList right) {
    assertAlive();
    if (1 < currentRefCount()) {
      TensorList sum = add(right);
      freeRef();
      return sum;
    }
    assert length() == right.length();
    if (heapCopy == null) {
      if (right instanceof GpuTensorList) {
        @javax.annotation.Nonnull final GpuTensorList nativeRight = (GpuTensorList) right;
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
    assert length() == right.length();
    if (heapCopy == null) {
      if (right instanceof GpuTensorList) {
        @javax.annotation.Nonnull final GpuTensorList nativeRight = (GpuTensorList) right;
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
  
  private GpuTensorList gpuAddInPlace(final GpuTensorList nativeRight) {
    return GpuSystem.eval(gpu -> {
      assert getDimensions().length <= 3;
      int d2 = getDimensions().length < 3 ? 1 : getDimensions()[2];
      int d1 = getDimensions().length < 2 ? 1 : getDimensions()[1];
      int d0 = getDimensions()[0];
      @Nonnull CudaPtr rPtr = nativeRight.getPtr();
      @Nonnull CudaPtr lPtr = GpuTensorList.this.getPtr();
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
        this.precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, getLength(), d2, d1, d0);
      gpu.cudnnAddTensor(
        precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr()
      );
      gpu.registerForCleanup(sizeDescriptor, rPtr, lPtr);
      return GpuTensorList.this;
    });
  }
  
  private GpuTensorList gpuAdd(final GpuTensorList nativeRight) {
    return GpuSystem.eval(gpu -> {
      assert getDimensions().length <= 3;
      int d2 = getDimensions().length < 3 ? 1 : getDimensions()[2];
      int d1 = getDimensions().length < 2 ? 1 : getDimensions()[1];
      int d0 = getDimensions()[0];
      @Nonnull CudaPtr rPtr = nativeRight.getPtr();
      @Nonnull CudaPtr lPtr = GpuTensorList.this.getPtr();
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = GpuSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, this.precision.code);
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
        this.precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, getLength(), d2, d1, d0);
      @Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
      gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
        precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
      gpu.registerForCleanup(opDescriptor, sizeDescriptor, rPtr, lPtr);
      return GpuTensorList.wrap(outputPtr, getLength(), getDimensions(), this.precision);
    });
  }
  
  @Override
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
    return GpuSystem.eval(gpu -> {
      CudaPtr ptr = getPtr();
      CudaPtr copyPtr = ptr.copyTo(gpu.getDeviceNumber());
      ptr.freeRef();
      @javax.annotation.Nonnull GpuTensorList gpuTensorList = new GpuTensorList(copyPtr, getLength(), getDimensions(), precision);
      copyPtr.freeRef();
      return gpuTensorList;
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
      }
    }
  }
  
  /**
   * Evict to heap.
   */
  public synchronized void evictToHeap() {
    synchronized (this) {
      if (null == getHeapCopy()) {
        throw new IllegalStateException();
      }
      if (null != ptr && !ptr.isFinalized()) {
        ptr.freeRef();
        ptr = null;
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
