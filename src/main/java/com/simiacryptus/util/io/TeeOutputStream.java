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

import javax.annotation.Nullable;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The type Tee output stream.
 */
public class TeeOutputStream extends OutputStream {
  /**
   * The Branches.
   */
  public final List<OutputStream> branches = new ArrayList<>();
  /**
   * The Primary.
   */
  public final OutputStream primary;
  @Nullable
  private final ByteArrayOutputStream heapBuffer;
  
  /**
   * Instantiates a new Tee output stream.
   *
   * @param primary the primary
   * @param buffer  the buffer
   */
  public TeeOutputStream(final OutputStream primary, final boolean buffer) {
    this.primary = primary;
    if (buffer) {
      heapBuffer = new ByteArrayOutputStream();
      branches.add(heapBuffer);
    }
    else {
      heapBuffer = null;
    }
  }
  
  @Override
  public void close() throws IOException {
    primary.close();
    for (@javax.annotation.Nonnull final OutputStream branch : branches) {
      branch.close();
    }
  }
  
  @Override
  public void flush() throws IOException {
    primary.flush();
    for (@javax.annotation.Nonnull final OutputStream branch : branches) {
      branch.flush();
    }
  }
  
  /**
   * New input stream piped input stream.
   *
   * @return the piped input stream
   * @throws IOException the io exception
   */
  @javax.annotation.Nonnull
  public PipedInputStream newInputStream() throws IOException {
    @javax.annotation.Nonnull final TeeOutputStream outTee = this;
    @javax.annotation.Nonnull final AtomicReference<Runnable> onClose = new AtomicReference<>();
    @javax.annotation.Nonnull final PipedOutputStream outPipe = new PipedOutputStream();
    @javax.annotation.Nonnull final PipedInputStream in = new PipedInputStream() {
      @Override
      public void close() throws IOException {
        outPipe.close();
        super.close();
      }
    };
    outPipe.connect(in);
    @javax.annotation.Nonnull final OutputStream outAsync = new AsyncOutputStream(outPipe);
    new Thread(() -> {
      try {
        if (null != heapBuffer) {
          outAsync.write(heapBuffer.toByteArray());
          outAsync.flush();
        }
        outTee.branches.add(outAsync);
      } catch (@javax.annotation.Nonnull final IOException e) {
        e.printStackTrace();
      }
    }).start();
    onClose.set(() -> {
      outTee.branches.remove(outAsync);
      System.err.println("END HTTP Session");
    });
    return in;
  }
  
  @Override
  public synchronized void write(final byte[] b) throws IOException {
    primary.write(b);
    for (@javax.annotation.Nonnull final OutputStream branch : branches) {
      branch.write(b);
    }
  }
  
  @Override
  public synchronized void write(final byte[] b, final int off, final int len) throws IOException {
    primary.write(b, off, len);
    for (@javax.annotation.Nonnull final OutputStream branch : branches) {
      branch.write(b, off, len);
    }
  }
  
  @Override
  public synchronized void write(final int b) throws IOException {
    primary.write(b);
    for (@javax.annotation.Nonnull final OutputStream branch : branches) {
      branch.write(b);
    }
  }
}
