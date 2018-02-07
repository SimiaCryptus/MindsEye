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

package com.simiacryptus.util.io;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The type Binary chunk iterator.
 */
public final class BinaryChunkIterator implements Iterator<byte[]> {
  
  private final DataInputStream in;
  private final int recordSize;
  
  /**
   * Instantiates a new Binary chunk iterator.
   *
   * @param in         the in
   * @param recordSize the record size
   */
  public BinaryChunkIterator(final DataInputStream in, final int recordSize) {
    super();
    this.in = in;
    this.recordSize = recordSize;
  }
  
  @javax.annotation.Nonnull
  private static byte[] read(@javax.annotation.Nonnull final DataInputStream i, final int s) throws IOException {
    @javax.annotation.Nonnull final byte[] b = new byte[s];
    int pos = 0;
    while (b.length > pos) {
      final int read = i.read(b, pos, b.length - pos);
      if (0 == read) {
        throw new RuntimeException();
      }
      pos += read;
    }
    return b;
  }
  
  /**
   * To iterator stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @return the stream
   */
  public static <T> Stream<T> toIterator(@javax.annotation.Nonnull final Iterator<T> iterator) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, 1, Spliterator.ORDERED), false);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @return the stream
   */
  public static <T> Stream<T> toStream(@javax.annotation.Nonnull final Iterator<T> iterator) {
    return BinaryChunkIterator.toStream(iterator, 0);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @param size     the size
   * @return the stream
   */
  public static <T> Stream<T> toStream(@javax.annotation.Nonnull final Iterator<T> iterator, final int size) {
    return BinaryChunkIterator.toStream(iterator, size, false);
  }
  
  /**
   * To stream stream.
   *
   * @param <T>      the type parameter
   * @param iterator the iterator
   * @param size     the size
   * @param parallel the parallel
   * @return the stream
   */
  public static <T> Stream<T> toStream(@javax.annotation.Nonnull final Iterator<T> iterator, final int size, final boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }
  
  @Override
  public boolean hasNext() {
    try {
      return 0 < in.available();
    } catch (@javax.annotation.Nonnull final Throwable e) {
      return false;
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public byte[] next() {
    assert hasNext();
    try {
      return BinaryChunkIterator.read(in, recordSize);
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * To stream stream.
   *
   * @return the stream
   */
  public Stream<byte[]> toStream() {
    return BinaryChunkIterator.toStream(this);
  }
}
