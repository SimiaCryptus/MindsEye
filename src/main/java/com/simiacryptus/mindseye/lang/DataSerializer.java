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

import org.jetbrains.annotations.NotNull;

/**
 * Provides a data serialization interface designed for converting arrays of doubles to/from arrays of bytes.
 * Implementations may use reduced precision and other lossy compression techniques.
 */
public interface DataSerializer {
  
  /**
   * Copy.
   *
   * @param from the from
   * @param to   the to
   */
  void copy(double[] from, byte[] to);
  
  /**
   * Copy.
   *
   * @param from the from
   * @param to   the to
   */
  void copy(byte[] from, double[] to);
  
  /**
   * Gets element size.
   *
   * @return the element size
   */
  int getElementSize();
  
  /**
   * To bytes byte [ ].
   *
   * @param from the from
   * @return the byte [ ]
   */
  default @NotNull byte[] toBytes(@NotNull double[] from) {
    @NotNull byte[] to = new byte[encodedSize(from)];
    copy(from, to);
    return to;
  }
  
  /**
   * Encoded size int.
   *
   * @param from the from
   * @return the int
   */
  default int encodedSize(@NotNull double[] from) {
    return from.length * getElementSize() + getHeaderSize();
  }
  
  /**
   * Gets header size.
   *
   * @return the header size
   */
  default int getHeaderSize() {
    return 0;
  }
  
  /**
   * From bytes double [ ].
   *
   * @param from the from
   * @return the double [ ]
   */
  default @NotNull double[] fromBytes(@NotNull byte[] from) {
    @NotNull double[] to = new double[decodedSize(from)];
    copy(from, to);
    return to;
  }
  
  /**
   * Decoded size int.
   *
   * @param from the from
   * @return the int
   */
  default int decodedSize(@NotNull byte[] from) {
    return (from.length - getHeaderSize()) / getElementSize();
  }
}
