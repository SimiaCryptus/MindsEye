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

import com.simiacryptus.mindseye.lang.cudnn.Precision;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This abstract data container is used to pass data between LayerBase components. It potentially represents data stored
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
  default TensorList add(@Nonnull final TensorList right) {
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length())
      .mapToObj(i -> {
        Tensor b = right.get(i);
        @Nullable Tensor sum = get(i).addAndFree(b);
        b.freeRef();
        return sum;
      })
      .toArray(i -> new Tensor[i]));
  }
  
  /**
   * Add and free tensor list.
   *
   * @param right the right
   * @return the tensor list
   */
  default TensorList addAndFree(@Nonnull final TensorList right) {
    assertAlive();
    right.assertAlive();
    TensorList add = add(right);
    freeRef();
    return add;
  }
  
  /**
   * Minus tensor list.
   *
   * @param right the right
   * @return the tensor list
   */
  @Nonnull
  default TensorList minus(@Nonnull final TensorList right) {
    if (right.length() == 0) return this;
    if (length() == 0) throw new IllegalArgumentException();
    assert length() == right.length();
    return TensorArray.wrap(IntStream.range(0, length()).mapToObj(i -> {
      @Nullable Tensor a = get(i);
      @Nullable Tensor b = right.get(i);
      @Nonnull Tensor r = a.minus(b);
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
        @Nullable Tensor element = get(i);
        @Nonnull Tensor copy = element.copy();
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
  @Nonnull
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
  @Nonnull
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
  @Nonnull
  default Tensor getAndFree(int i) {
    @Nullable Tensor tensor = get(i);
    freeRef();
    return tensor;
  }
  
  /**
   * Gets elements.
   *
   * @return the elements
   */
  default int getElements() {
    return length() * Tensor.length(getDimensions());
  }
  
  /**
   * Gets bytes.
   *
   * @param precision the precision
   * @return the bytes
   */
  default int getBytes(Precision precision) {
    return length() * Tensor.length(getDimensions()) * precision.size;
  }
}
