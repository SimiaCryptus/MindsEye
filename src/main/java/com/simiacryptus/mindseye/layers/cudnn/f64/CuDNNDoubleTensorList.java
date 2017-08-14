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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.simiacryptus.mindseye.layers.TensorArray;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Cu dnn double tensor list.
 */
public class CuDNNDoubleTensorList implements TensorList {
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
   * Instantiates a new Cu dnn double tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   */
  public CuDNNDoubleTensorList(CudaPtr ptr, int length, int[] dimensions) {
          this.ptr = ptr;
          this.length = length;
          this.dimensions = dimensions;
          assert(ptr.size == length * Tensor.dim(dimensions) * Sizeof.DOUBLE);
          //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      }

      private volatile TensorList _inner = null;

  /**
   * Inner tensor list.
   *
   * @return the tensor list
   */
  public TensorList inner() {
          if(null == _inner) {
              synchronized (this) {
                  if(null == _inner) {
                      int itemLength = Tensor.dim(dimensions);
                      final double[] outputBuffer = Tensor.obtain(itemLength * length);
                      assert(0 < outputBuffer.length);
                      Tensor[] output = IntStream.range(0, length)
                              .mapToObj(dataIndex -> new Tensor(dimensions))
                              .toArray(i -> new Tensor[i]);
                      double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
                      assert(length == outputBuffers.length);
                      ptr.read(outputBuffer);
                      for (int i = 0; i< length; i++) {
                        assert itemLength == outputBuffers[0 +i].length;
                        System.arraycopy(outputBuffer, i * itemLength, outputBuffers[0 +i], 0, itemLength);
                      }
                      //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                      Tensor.recycle(outputBuffer);
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
}
