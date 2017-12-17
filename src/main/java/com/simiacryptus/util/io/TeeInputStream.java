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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


/**
 * The type Tee input stream.
 */
public class TeeInputStream extends InputStream {
  
  private final OutputStream cache;
  private final InputStream inputStream;
  
  
  /**
   * Instantiates a new Tee input stream.
   *
   * @param inputStream the input stream
   * @param cache       the cache
   */
  public TeeInputStream(final InputStream inputStream, final OutputStream cache) {
    this.inputStream = inputStream;
    this.cache = cache;
  }
  
  @Override
  public int available() throws IOException {
    return inputStream.available();
  }
  
  @Override
  public void close() throws IOException {
    inputStream.close();
    cache.close();
  }
  
  @Override
  public int read() throws IOException {
    final int read = inputStream.read();
    cache.write(read);
    return read;
  }
  
  @Override
  public int read(final byte[] b) throws IOException {
    final int read = inputStream.read(b);
    if (read > 0) {
      cache.write(b);
    }
    return read;
  }
  
  @Override
  public int read(final byte[] b, final int off, final int len) throws IOException {
    final int read = inputStream.read(b, off, len);
    if (read > 0) {
      cache.write(b, off, read);
    }
    return read;
  }
}
