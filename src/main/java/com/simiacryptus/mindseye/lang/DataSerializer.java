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
  @Nonnull
  default byte[] toBytes(@Nonnull double[] from) {
    @Nonnull byte[] to = new byte[encodedSize(from)];
    copy(from, to);
    return to;
  }
  
  /**
   * Encoded size int.
   *
   * @param from the from
   * @return the int
   */
  default int encodedSize(@Nonnull double[] from) {
    long size = (long) from.length * getElementSize() + getHeaderSize();
    if (size > Integer.MAX_VALUE) throw new IllegalStateException();
    return (int) size;
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
  @Nonnull
  default double[] fromBytes(@Nonnull byte[] from) {
    @Nonnull double[] to = new double[decodedSize(from)];
    copy(from, to);
    return to;
  }
  
  /**
   * Decoded size int.
   *
   * @param from the from
   * @return the int
   */
  default int decodedSize(@Nonnull byte[] from) {
    return (from.length - getHeaderSize()) / getElementSize();
  }
}
