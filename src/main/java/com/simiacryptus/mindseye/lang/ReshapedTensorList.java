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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.stream.Stream;

/**
 * A wrapper TensorList data to override the existing tensor layer. Can be used for example to flatten or unflatten a
 * tensor to/from a rank-1 array.
 */
public class ReshapedTensorList extends ReferenceCountingBase implements TensorList {
  @javax.annotation.Nonnull
  private final TensorList inner;
  private final int[] dims;
  
  /**
   * Instantiates a new Reshaped tensor list.
   *
   * @param inner  the data
   * @param toDim the to dim
   */
  public ReshapedTensorList(@javax.annotation.Nonnull TensorList inner, int[] toDim) {
    if (Tensor.dim(inner.getDimensions()) != Tensor.dim(toDim))
      throw new IllegalArgumentException(Arrays.toString(inner.getDimensions()) + " != " + Arrays.toString(toDim));
    this.inner = inner;
    this.inner.addRef(this);
    this.dims = toDim;
  }
  
  @Nullable
  @Override
  public Tensor get(int i) {
    assertAlive();
    @javax.annotation.Nullable Tensor tensor = inner.get(i);
    @javax.annotation.Nullable Tensor reshapeCast = tensor.reshapeCast(dims);
    tensor.freeRef();
    return reshapeCast;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[] getDimensions() {
    return Arrays.copyOf(dims, dims.length);
  }
  
  @Override
  public int length() {
    return inner.length();
  }
  
  @Override
  public Stream<Tensor> stream() {
    return inner.stream().map(t -> {
      @javax.annotation.Nullable Tensor tensor = t.reshapeCast(dims);
      t.freeRef();
      return tensor;
    });
  }
  
  @Override
  protected void _free() {
    inner.freeRef();
  }
  
  @Nonnull
  public TensorList getInner() {
    return inner;
  }
}
