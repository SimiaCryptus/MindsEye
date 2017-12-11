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

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The type Tee output stream.
 */
public class TeeOutputStream extends OutputStream {
  /**
   * The Primary.
   */
  public final OutputStream primary;
  /**
   * The Branches.
   */
  public final List<OutputStream> branches = new ArrayList<>();
  
  private final ByteArrayOutputStream heapBuffer;
  
  /**
   * Instantiates a new Tee output stream.
   *
   * @param primary the primary
   * @param buffer  the buffer
   */
  public TeeOutputStream(OutputStream primary, boolean buffer) {
    this.primary = primary;
    if (buffer) {
      heapBuffer = new ByteArrayOutputStream();
      branches.add(heapBuffer);
    }
    else {
      heapBuffer = null;
    }
  }
  
  public synchronized void write(byte[] b) throws IOException {
    this.primary.write(b);
    for (OutputStream branch : this.branches) branch.write(b);
  }
  
  public synchronized void write(byte[] b, int off, int len) throws IOException {
    this.primary.write(b, off, len);
    for (OutputStream branch : this.branches) branch.write(b, off, len);
  }
  
  public synchronized void write(int b) throws IOException {
    this.primary.write(b);
    for (OutputStream branch : this.branches) branch.write(b);
  }
  
  public void flush() throws IOException {
    this.primary.flush();
    for (OutputStream branch : this.branches) branch.flush();
  }
  
  public void close() throws IOException {
    this.primary.close();
    for (OutputStream branch : this.branches) branch.close();
  }
  
  /**
   * New input stream piped input stream.
   *
   * @return the piped input stream
   * @throws IOException the io exception
   */
  public PipedInputStream newInputStream() throws IOException {
    TeeOutputStream outTee = this;
    final AtomicReference<Runnable> onClose = new AtomicReference<>();
    final PipedOutputStream outPipe = new PipedOutputStream();
    PipedInputStream in = new PipedInputStream() {
      @Override
      public void close() throws IOException {
        outPipe.close();
        super.close();
      }
    };
    outPipe.connect(in);
    OutputStream outAsync = new AsyncOutputStream(outPipe);
    new Thread(() -> {
      try {
        if (null != heapBuffer) {
          outAsync.write(heapBuffer.toByteArray());
          outAsync.flush();
        }
        outTee.branches.add(outAsync);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }).start();
    onClose.set(() -> {
      outTee.branches.remove(outAsync);
      System.err.println("END HTTP Session");
    });
    return in;
  }
}
