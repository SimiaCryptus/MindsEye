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

/**
 * A data structure to represent an index/coordinate/tuple for referencing elements in a Tensor.
 * It contains both the physical (1-d) and logical (N-d) indicies of the element.
 */
public final class Coordinate {
  private int[] coords;
  private int index;
  
  /**
   * Instantiates a new Coordinate.
   */
  public Coordinate() {
    this(-1, null);
  }
  
  /**
   * Instantiates a new Coordinate.
   *
   * @param index  the index
   * @param coords the coords
   */
  public Coordinate(final int index, final int[] coords) {
    super();
    this.index = index;
    this.coords = coords;
  }
  
  /**
   * Add int [ ].
   *
   * @param a the a
   * @param b the b
   * @return the int [ ]
   */
  public static int[] add(final int[] a, final int[] b) {
    final int[] r = new int[Math.max(a.length, b.length)];
    for (int i = 0; i < r.length; i++) {
      r[i] = (a.length <= i ? 0 : a[i]) + (b.length <= i ? 0 : b[i]);
    }
    return r;
  }
  
  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final Coordinate other = (Coordinate) obj;
    return Arrays.equals(this.coords, other.coords);
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(this.coords);
    return result;
  }
  
  @Override
  public String toString() {
    return Arrays.toString(this.coords) + "<" + this.index + ">";
  }
  
  /**
   * The Coords.
   *
   * @return the int [ ]
   */
  public int[] getCoords() {
    return coords;
  }
  
  /**
   * Sets coords.
   *
   * @param coords the coords
   * @return the coords
   */
  Coordinate setCoords(int[] coords) {
    this.coords = coords;
    return this;
  }
  
  /**
   * The Index.
   *
   * @return the index
   */
  public int getIndex() {
    return index;
  }
  
  /**
   * Sets index.
   *
   * @param index the index
   * @return the index
   */
  Coordinate setIndex(int index) {
    this.index = index;
    return this;
  }
}
