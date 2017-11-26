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

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * An on-heap implementation of the TensorList data container.
 */
public class TensorArray implements TensorList {
  private final Tensor[] data;
  
  /**
   * Instantiates a new Tensor array.
   *
   * @param data the data
   */
  public TensorArray(Tensor... data) {
    this.data = data;
  }
  
  @Override
  public Tensor get(int i) {
    return data[i];
  }
  
  @Override
  public int length() {
    return data.length;
  }
  
  @Override
  public Stream<Tensor> stream() {
    return Arrays.stream(data);
  }
  
  @Override
  public int[] getDimensions() {
    return data[0].getDimensions();
  }
  
}
