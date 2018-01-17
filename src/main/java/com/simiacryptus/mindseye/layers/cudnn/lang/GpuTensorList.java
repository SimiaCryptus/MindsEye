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

package com.simiacryptus.mindseye.layers.cudnn.lang;

import com.simiacryptus.mindseye.lang.*;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A TensorList data object stored on a GPU with a configurable precision.
 */
public class GpuTensorList implements TensorList {
  protected static final Logger logger = LoggerFactory.getLogger(GpuTensorList.class);
  public static boolean DISABLE_GPU_INTERCONNECT = true;
  
  /**
   * The Dimensions.
   */
  public final int[] dimensions;
  /**
   * The Length.
   */
  public final int length;
  /**
   * The Ptr.
   */
  public final ManagedCudaPtr ptr;
  private final Precision precision;
  private volatile TensorList _inner = null;
  
  /**
   * Instantiates a new Cu dnn double tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @param precision  the precision
   */
  public GpuTensorList(final CudaPtr ptr, final int length, final int[] dimensions, final Precision precision) {
    this.precision = precision;
    if (null == ptr) throw new IllegalArgumentException("ptr");
    if (null == ptr.getPtr()) throw new IllegalArgumentException("ptr.getPtr()");
    this.ptr = ptr.managed();
    this.length = length;
    this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
    assert ptr.size == length * Tensor.dim(dimensions) * precision.size;
    assert ptr.getPtr() != null;
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    assert !System.getProperties().containsKey("safe") || stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
  }
  
  @Override
  public synchronized void addInPlace(final TensorList right) {
    assert length() == right.length();
    if (_inner == null) {
      if (right instanceof GpuTensorList) {
        final GpuTensorList nativeRight = (GpuTensorList) right;
        if (nativeRight.precision == precision) {
          if (nativeRight._inner == null) {
            if (CuDNN.run(exe -> {
              if (DISABLE_GPU_INTERCONNECT) return false;
              if (nativeRight.ptr.getDeviceId() != GpuTensorList.this.ptr.getDeviceId()) return false;
              final CudaResource<cudnnTensorDescriptor> leftSize = CuDNN.newTensorDescriptor(precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
              final CudaResource<cudnnTensorDescriptor> rightSize = CuDNN.newTensorDescriptor(precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
              CuDNN.handle(CuDNN.cudnnAddTensor(exe.cudnnHandle,
                                                precision.getPointer(1.0), rightSize.getPtr(), nativeRight.ptr.getPtr(),
                                                precision.getPointer(1.0), leftSize.getPtr(), GpuTensorList.this.ptr.getPtr()));
              leftSize.finalize();
              rightSize.finalize();
              return true;
            })) return;
          }
        }
      }
    }
    IntStream.range(0, length()).forEach(i -> {
      get(i).accumulate(right.get(i));
    });
  }
  
  @Override
  public Tensor get(final int i) {
    return localCopy().get(i);
  }
  
  @Override
  public int[] getDimensions() {
    return Arrays.copyOf(dimensions, dimensions.length);
  }
  
  /**
   * Gets precision.
   *
   * @return the precision
   */
  public Precision getPrecision() {
    return precision;
  }
  
  /**
   * Inner tensor list.
   *
   * @return the tensor list
   */
  public TensorList localCopy() {
    if (null == _inner) {
      synchronized (this) {
        if (null == _inner) {
          final int itemLength = Tensor.dim(dimensions);
          final double[] outputBuffer = RecycleBin.DOUBLES.obtain(itemLength * length);
          assert 0 < outputBuffer.length;
          final Tensor[] output = IntStream.range(0, length)
                                           .mapToObj(dataIndex -> new Tensor(dimensions))
                                           .toArray(i -> new Tensor[i]);
          final double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
          assert length == outputBuffers.length;
          ptr.read(precision, outputBuffer);
          for (int i = 0; i < length; i++) {
            assert itemLength == outputBuffers[0 + i].length;
            System.arraycopy(outputBuffer, i * itemLength, outputBuffers[0 + i], 0, itemLength);
          }
          //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
          _inner = new TensorArray(output);
        }
      }
    }
    return _inner;
  }
  
  /**
   * Sets gpu persistance.
   *
   * @param persistanceMode the persistance mode
   * @return the gpu persistance
   */
  public TensorList setGpuPersistance(PersistanceMode persistanceMode) {
    ptr.setGpuPersistance(persistanceMode);
    return this;
  }
  
  @Override
  public int length() {
    return length;
  }
  
  @Override
  public void recycle() {
    ptr.free();
    if (null != _inner) {
      _inner.recycle();
    }
  }
  
  @Override
  public Stream<Tensor> stream() {
    return localCopy().stream();
  }
  
  public boolean isNative() {
    return null == _inner;
  }
  
}
