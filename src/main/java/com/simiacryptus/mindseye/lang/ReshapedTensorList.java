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

package com.simiacryptus.mindseye.lang;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * The type Reshaped tensor list.
 */
public class ReshapedTensorList implements TensorList {
  private final TensorList data;
  private final int[] dims;
  
  /**
   * Instantiates a new Reshaped tensor list.
   *
   * @param data  the data
   * @param toDim the to dim
   */
  public ReshapedTensorList(TensorList data, int[] toDim) {
    if (Tensor.dim(data.getDimensions()) != Tensor.dim(toDim))
      throw new IllegalArgumentException(Arrays.toString(data.getDimensions()) + " != " + Arrays.toString(toDim));
    this.data = data;
    this.dims = toDim;
  }
  
  @Override
  public Tensor get(int i) {
    return data.get(i).reshapeCast(dims);
  }
  
  @Override
  public int[] getDimensions() {
    return Arrays.copyOf(dims, dims.length);
  }
  
  @Override
  public int length() {
    return data.length();
  }
  
  @Override
  public void recycle() {
    data.recycle();
  }
  
  @Override
  public Stream<Tensor> stream() {
    return data.stream().map(t -> t.reshapeCast(dims));
  }
}
