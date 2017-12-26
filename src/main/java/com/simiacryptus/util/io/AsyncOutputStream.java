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

package com.simiacryptus.util.io;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * The type Async output stream.
 */
public class AsyncOutputStream extends FilterOutputStream {
  
  private final FairAsyncWorkQueue queue = new FairAsyncWorkQueue();
  
  /**
   * Instantiates a new Async output stream.
   *
   * @param stream the stream
   */
  public AsyncOutputStream(final OutputStream stream) {
    super(stream);
  }
  
  @Override
  public synchronized void close() throws IOException {
    queue.submit(() -> {
      try {
        out.close();
      } catch (final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  @Override
  public synchronized void flush() throws IOException {
    queue.submit(() -> {
      try {
        out.flush();
      } catch (final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  @Override
  public synchronized void write(final byte[] b, final int off, final int len) throws IOException {
    final byte[] _b = Arrays.copyOfRange(b, off, Math.min(b.length, off + len));
    queue.submit(() -> {
      try {
        out.write(_b);
      } catch (final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  @Override
  public synchronized void write(final int b) throws IOException {
    queue.submit(() -> {
      try {
        out.write(b);
      } catch (final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
}
