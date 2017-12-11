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

import com.simiacryptus.util.lang.UncheckedSupplier;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * The interface Notebook output.
 */
public interface NotebookOutput extends Closeable {
  
  /**
   * Out.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  default void out(String fmt, Object... args) {
    p(fmt, args);
  }
  
  /**
   * P.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  void p(String fmt, Object... args);
  
  /**
   * H 1.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  void h1(String fmt, Object... args);
  
  /**
   * H 2.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  void h2(String fmt, Object... args);
  
  /**
   * H 3.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  void h3(String fmt, Object... args);
  
  /**
   * Code t.
   *
   * @param <T>      the type parameter
   * @param fn       the fn
   * @param maxLog   the max log
   * @param framesNo the frames no
   * @return the t
   */
  <T> T code(UncheckedSupplier<T> fn, int maxLog, int framesNo);
  
  /**
   * File string.
   *
   * @param data    the data
   * @param caption the caption
   * @return the string
   */
  String file(String data, String caption);
  
  /**
   * File string.
   *
   * @param data     the data
   * @param fileName the file name
   * @param caption  the caption
   * @return the string
   */
  String file(String data, String fileName, String caption);
  
  /**
   * Image string.
   *
   * @param rawImage the raw image
   * @param caption  the caption
   * @return the string
   * @throws IOException the io exception
   */
  String image(BufferedImage rawImage, String caption) throws IOException;
  
  /**
   * File output stream.
   *
   * @param name the name
   * @return the output stream
   */
  OutputStream file(String name);
  
  /**
   * Code t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  default <T> T code(UncheckedSupplier<T> fn) {
    return code(fn,1024, 3);
  }
  
  /**
   * Code.
   *
   * @param fn       the fn
   * @param maxLog   the max log
   * @param framesNo the frames no
   */
  default void code(Runnable fn, int maxLog, int framesNo) {
    this.code(() -> {
      fn.run();
      return null;
    }, maxLog, framesNo);
  }
  
  /**
   * Code.
   *
   * @param fn the fn
   */
  default void code(Runnable fn) {
    this.code(() -> {
      fn.run();
      return null;
    }, 1024, 3);
  }
  
  /**
   * Add copy notebook output.
   *
   * @param out the out
   * @return the notebook output
   */
  NotebookOutput addCopy(PrintStream out);
  
}
