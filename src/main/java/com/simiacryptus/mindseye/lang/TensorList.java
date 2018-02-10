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
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This abstract data container is used to pass data between NNLayer components. It potentially represents data stored
 * off-heap, such as on a particular GPU. Use of this abstract class allows optimizations where adjacent GPU components
 * can operate with minimal CPU-GPU data transfer.
 */
public interface TensorList extends ReferenceCounting {
  
  /**
   * Add tensor list.
   *
   * @param right the right
   * @return the tensor list
   */
  default TensorList add(@javax.annotation.Nonnull final TensorList right) {
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length())
      .mapToObj(i -> freeSum(get(i), right.get(i)))
      .toArray(i -> new Tensor[i]));
  }
  
  /**
   * Free sum tensor.
   *
   * @param a the a
   * @param b the b
   * @return the tensor
   */
  @Nullable
  default Tensor freeSum(@javax.annotation.Nonnull Tensor a, @javax.annotation.Nonnull Tensor b) {
    @Nullable Tensor sum = a.add(b);
    a.freeRef();
    b.freeRef();
    return sum;
  }
  
  /**
   * Minus tensor list.
   *
   * @param right the right
   * @return the tensor list
   */
  @Nonnull
  default TensorList minus(@javax.annotation.Nonnull final TensorList right) {
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      @javax.annotation.Nullable Tensor a = get(i);
      @javax.annotation.Nullable Tensor b = right.get(i);
      @javax.annotation.Nonnull Tensor r = a.minus(b);
      a.freeRef();
      b.freeRef();
      return r;
    }).toArray(i -> new Tensor[i]));
  }
  
  /**
   * Copy tensor list.
   *
   * @return the tensor list
   */
  default TensorList copy() {
    return TensorArray.wrap(
      IntStream.range(0, length()).mapToObj(i -> {
        @javax.annotation.Nullable Tensor element = get(i);
        @javax.annotation.Nonnull Tensor copy = element.copy();
        element.freeRef();
        return copy;
      }).toArray(i -> new Tensor[i])
    );
  }
  
  /**
   * Get tensor.
   *
   * @param i the
   * @return the tensor
   */
  @javax.annotation.Nullable
  Tensor get(int i);
  
  /**
   * Get dimensions int [ ].
   *
   * @return the int [ ]
   */
  @Nonnull
  int[] getDimensions();
  
  /**
   * Length int.
   *
   * @return the int
   */
  int length();
  
  /**
   * Stream stream.
   *
   * @return the stream
   */
  Stream<Tensor> stream();
  
  /**
   * Pretty print string.
   *
   * @return the string
   */
  @javax.annotation.Nonnull
  default String prettyPrint() {
    return stream().map(t -> {
      String str = t.prettyPrint();
      t.freeRef();
      return str;
    }).reduce((a, b) -> a + "\n" + b).get();
  }
  
  /**
   * Gets and free.
   *
   * @param i the
   * @return the and free
   */
  @javax.annotation.Nullable
  default Tensor getAndFree(int i) {
    @javax.annotation.Nullable Tensor tensor = get(i);
    freeRef();
    return tensor;
  }
}
