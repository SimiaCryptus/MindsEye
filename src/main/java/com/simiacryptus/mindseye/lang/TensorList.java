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

package com.simiacryptus.mindseye.lang;

import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The interface Tensor list.
 */
public interface TensorList {
  /**
   * Get tensor.
   *
   * @param i the
   * @return the tensor
   */
  Tensor get(int i);
  
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
   * Get dimensions int [ ].
   *
   * @return the int [ ]
   */
  int[] getDimensions();
  
  /**
   * Add tensor list.
   *
   * @param right the right
   * @return the tensor list
   */
  default TensorList add(TensorList right) {
    assert (length() == right.length());
    return new TensorArray(
      IntStream.range(0, length()).mapToObj(i -> {
        return get(i).add(right.get(i));
      }).toArray(i -> new Tensor[i])
    );
  }
  
  /**
   * Accum.
   *
   * @param right the right
   */
  default void accum(TensorList right) {
    if (right.length() == 0) return;
    if (this.length() == 0) throw new IllegalArgumentException();
    assert (length() == right.length());
    IntStream.range(0, length()).forEach(i -> {
      get(i).accum(right.get(i));
    });
  }
  
  /**
   * Copy tensor list.
   *
   * @return the tensor list
   */
  default TensorList copy() {
    return new TensorArray(
      IntStream.range(0, length()).mapToObj(i -> get(i).copy()).toArray(i -> new Tensor[i])
    );
  }
  
}
