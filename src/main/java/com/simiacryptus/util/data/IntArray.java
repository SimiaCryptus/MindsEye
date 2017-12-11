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

package com.simiacryptus.util.data;

import java.util.Arrays;

/**
 * The type Int array.
 */
public class IntArray {
  
  /**
   * The Data.
   */
  public final int[] data;
  
  /**
   * Instantiates a new Int array.
   *
   * @param data the data
   */
  public IntArray(int[] data) {
    this.data = data;
  }
  
  /**
   * Get int.
   *
   * @param i the
   * @return the int
   */
  public int get(int i) {
    return data[i];
  }
  
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    
    IntArray intArray = (IntArray) o;

    return Arrays.equals(data, intArray.data);
  }
  
  @Override
  public int hashCode() {
    return Arrays.hashCode(data);
  }
  
  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return data.length;
  }
}
