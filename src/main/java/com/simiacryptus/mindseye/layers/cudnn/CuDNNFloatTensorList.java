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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Cu dnn float tensor list.
 */
public class CuDNNFloatTensorList implements TensorList {
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
  private final jcuda.jcudnn.cudnnHandle cudnnHandle;
  private volatile TensorList _inner = null;
  
  /**
   * Instantiates a new Cu dnn float tensor list.
   *
   * @param ptr         the ptr
   * @param length      the length
   * @param dimensions  the dimensions
   * @param cudnnHandle the cudnn handle
   */
  public CuDNNFloatTensorList(CudaPtr ptr, int length, int[] dimensions, jcuda.jcudnn.cudnnHandle cudnnHandle) {
    this.ptr = ptr;
    this.length = length;
    this.dimensions = dimensions;
    this.cudnnHandle = cudnnHandle;
    assert (ptr.size == this.length * 1l * Tensor.dim(this.dimensions) * Sizeof.FLOAT);
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
          int itemLength = Tensor.dim(this.dimensions);
          final float[] buffer = new float[itemLength * this.length];
          assert (0 < buffer.length);
          assert (ptr.size == this.length * 1l * itemLength * Sizeof.FLOAT);
          
          //Arrays.stream(output).mapCoords(x -> x.getDataAsFloats()).toArray(i -> new float[i][]);
          ptr.read(buffer);
          //assert IntStream.range(0,buffer.length).mapToDouble(ii->buffer[ii]).allMatch(Double::isFinite);
          float[][] floats = IntStream.range(0, length)
            .mapToObj(dataIndex -> new float[itemLength])
            .toArray(i -> new float[i][]);
          for (int i = 0; i < length; i++) {
            assert itemLength == floats[0 + i].length;
            System.arraycopy(buffer, i * itemLength, floats[0 + i], 0, itemLength);
          }
          //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          Tensor[] output = Arrays.stream(floats).map(floats2 -> {
            return new Tensor(floats2, dimensions);
          }).toArray(i -> new Tensor[i]);
          _inner = new TensorArray(output);
        }
      }
    }
    return _inner;
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
  
  @Override
  public TensorList add(TensorList right) {
    assert (length() == right.length());
    if (right instanceof CuDNNFloatTensorList) {
      CuDNNFloatTensorList nativeRight = (CuDNNFloatTensorList) right;
      assert (cudnnHandle == nativeRight.cudnnHandle);
      CudaResource<cudnnTensorDescriptor> size = CuDNN.newTensorDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
      CuDNN.handle(cudnnAddTensor(cudnnHandle,
        Pointer.to(new float[]{1.0f}), size.getPtr(), nativeRight.ptr.getPtr(),
        Pointer.to(new float[]{1.0f}), size.getPtr(), CuDNNFloatTensorList.this.ptr.getPtr()));
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
    if (right instanceof CuDNNFloatTensorList) {
      CuDNNFloatTensorList nativeRight = (CuDNNFloatTensorList) right;
      assert (cudnnHandle == nativeRight.cudnnHandle);
      CudaResource<cudnnTensorDescriptor> size = CuDNN.newTensorDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length(), dimensions[2], dimensions[1], dimensions[0]);
      CuDNN.handle(cudnnAddTensor(cudnnHandle,
        Pointer.to(new float[]{1.0f}), size.getPtr(), nativeRight.ptr.getPtr(),
        Pointer.to(new float[]{1.0f}), size.getPtr(), CuDNNFloatTensorList.this.ptr.getPtr()));
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
  public TensorList copy() {
    return new CuDNNFloatTensorList(ptr.copy(), length, dimensions, cudnnHandle);
  }
}
