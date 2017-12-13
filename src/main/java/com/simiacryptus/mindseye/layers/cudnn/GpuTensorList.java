/*
 * Copyright (c) 2017 by Andrew Charneski.
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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.DoubleArrays;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnAddTensor;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Cu dnn double tensor list.
 */
public class GpuTensorList implements TensorList {
  
  /**
   * The Ptr.
   */
  public final CudaPtr ptr;
  /**
   * The Length.
   */
  public final int length;
  /**
   * The Dimensions.
   */
  public final int[] dimensions;
  /**
   * The Cudnn handle.
   */
  protected final jcuda.jcudnn.cudnnHandle cudnnHandle;
  private final Precision precision;
  private volatile TensorList _inner = null;
  
  /**
   * Instantiates a new Cu dnn double tensor list.
   *
   * @param ptr         the ptr
   * @param length      the length
   * @param dimensions  the dimensions
   * @param cudnnHandle the cudnn handle
   * @param precision   the precision
   */
  public GpuTensorList(CudaPtr ptr, int length, int[] dimensions, jcuda.jcudnn.cudnnHandle cudnnHandle, Precision precision) {
    this.precision = precision;
    if (null == ptr) throw new IllegalArgumentException("ptr");
    if (null == ptr.getPtr()) throw new IllegalArgumentException("ptr.getPtr()");
    this.ptr = ptr;
    this.length = length;
    this.dimensions = dimensions;
    this.cudnnHandle = cudnnHandle;
    assert (ptr.size == length * Tensor.dim(dimensions) * precision.size);
    assert ptr.getPtr() != null;
    //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    assert !System.getProperties().containsKey("safe") || this.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
  }
  
  /**
   * Inner tensor list.
   *
   * @return the tensor list
   */
  public TensorList inner() {
    if (null == _inner) {
      synchronized (this) {
        if (null == _inner) {
          int itemLength = Tensor.dim(dimensions);
          final double[] outputBuffer = DoubleArrays.obtain(itemLength * length);
          assert (0 < outputBuffer.length);
          Tensor[] output = IntStream.range(0, length)
            .mapToObj(dataIndex -> new Tensor(dimensions))
            .toArray(i -> new Tensor[i]);
          double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
          assert (length == outputBuffers.length);
          ptr.read(precision, outputBuffer);
          for (int i = 0; i < length; i++) {
            assert itemLength == outputBuffers[0 + i].length;
            System.arraycopy(outputBuffer, i * itemLength, outputBuffers[0 + i], 0, itemLength);
          }
          //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          DoubleArrays.recycle(outputBuffer);
          _inner = new TensorArray(output);
        }
      }
    }
    return _inner;
  }
  
  @Override
  public TensorList add(TensorList right) {
    assert (length() == right.length());
    if (right instanceof GpuTensorList && ((GpuTensorList) right).precision == precision) {
      GpuTensorList nativeRight = (GpuTensorList) right;
      assert (cudnnHandle == nativeRight.cudnnHandle);
      CudaResource<cudnnTensorDescriptor> size = CuDNN.newTensorDescriptor(precision.code, CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
      CuDNN.handle(cudnnAddTensor(cudnnHandle,
        precision.getPointer(1.0), size.getPtr(), nativeRight.ptr.getPtr(),
        precision.getPointer(1.0), size.getPtr(), GpuTensorList.this.ptr.getPtr()));
      size.finalize();
      nativeRight.ptr.finalize(); // Make this function destructive to both arguments
      return this;
    }
    return new TensorArray(
      IntStream.range(0, length()).mapToObj(i -> {
        return get(i).add(right.get(i));
      }).toArray(i -> new Tensor[i])
    );
  }
  
  @Override
  public void accum(TensorList right) {
    assert (length() == right.length());
    if (right instanceof GpuTensorList && ((GpuTensorList) right).precision == precision) {
      GpuTensorList nativeRight = (GpuTensorList) right;
      assert (cudnnHandle == nativeRight.cudnnHandle);
      CudaResource<cudnnTensorDescriptor> size = CuDNN.newTensorDescriptor(precision.code, CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
      CuDNN.handle(cudnnAddTensor(cudnnHandle,
        precision.getPointer(1.0), size.getPtr(), nativeRight.ptr.getPtr(),
        precision.getPointer(1.0), size.getPtr(), GpuTensorList.this.ptr.getPtr()));
      size.finalize();
      nativeRight.ptr.finalize(); // Make this function destructive to both arguments
    }
    else {
      IntStream.range(0, length()).forEach(i -> {
        get(i).accum(right.get(i));
      });
    }
  }
  
  @Override
  public Tensor get(int i) {
    return inner().get(i);
  }
  
  @Override
  public int length() {
    return length;
  }
  
  @Override
  public Stream<Tensor> stream() {
    return inner().stream();
  }
  
  @Override
  public int[] getDimensions() {
    return dimensions;
  }
  
  /**
   * Gets precision.
   *
   * @return the precision
   */
  public Precision getPrecision() {
    return precision;
  }
}
