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

import com.simiacryptus.util.lang.UncheckedSupplier;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;

/**
 * The interface Notebook output.
 */
public interface NotebookOutput extends Closeable {
  
  /**
   * Code.
   *
   * @param fn the fn
   */
  default void code(@javax.annotation.Nonnull final Runnable fn) {
    this.code(() -> {
      fn.run();
      return null;
    }, getMaxOutSize(), 3);
  }
  
  /**
   * Code.
   *
   * @param fn   the fn
   * @param size the size
   */
  default void code(@javax.annotation.Nonnull final Runnable fn, int size) {
    this.code(() -> {
      fn.run();
      return null;
    }, size, 3);
  }
  
  /**
   * Code.
   *
   * @param fn       the fn
   * @param maxLog   the max log
   * @param framesNo the frames no
   */
  default void code(@javax.annotation.Nonnull final Runnable fn, final int maxLog, final int framesNo) {
    this.code(() -> {
      fn.run();
      return null;
    }, maxLog, framesNo);
  }
  
  /**
   * Code t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  default <T> T code(final UncheckedSupplier<T> fn) {
    return code(fn, getMaxOutSize(), 3);
  }
  
  /**
   * Code t.
   *
   * @param <T>  the type parameter
   * @param fn   the fn
   * @param size the size
   * @return the t
   */
  default <T> T code(final UncheckedSupplier<T> fn, int size) {
    return code(fn, size, 3);
  }
  
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
   * File output stream.
   *
   * @param name the name
   * @return the output stream
   */
  @Nonnull
  OutputStream file(String name);
  
  /**
   * File string.
   *
   * @param data    the data
   * @param caption the caption
   * @return the string
   */
  @Nonnull
  String file(String data, String caption);
  
  /**
   * File string.
   *
   * @param data     the data
   * @param filename the filename
   * @param caption  the caption
   * @return the string
   */
  @Nonnull
  String file(byte[] data, String filename, String caption);
  
  /**
   * File string.
   *
   * @param data     the data
   * @param fileName the file name
   * @param caption  the caption
   * @return the string
   */
  @Nonnull
  String file(String data, String fileName, String caption);
  
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
   * Image string.
   *
   * @param rawImage the raw image
   * @param caption  the caption
   * @return the string
   * @throws IOException the io exception
   */
  @Nonnull
  String image(BufferedImage rawImage, String caption) throws IOException;
  
  /**
   * Link string.
   *
   * @param file the file
   * @param text the text
   * @return the string
   */
  String link(File file, String text);
  
  /**
   * Out.
   *
   * @param fmt  the fmt
   * @param args the args
   */
  default void out(final String fmt, final Object... args) {
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
   * Sets fm prop.
   *
   * @param key   the key
   * @param value the value
   */
  default void setFrontMatterProperty(String key, String value) {
  }
  
  /**
   * Append front matter property.
   *
   * @param key   the key
   * @param value the value
   */
  default void appendFrontMatterProperty(String key, String value) {appendFrontMatterProperty(key, value, "");}
  
  /**
   * Append front matter property.
   *
   * @param key       the key
   * @param value     the value
   * @param delimiter the delimiter
   */
  default void appendFrontMatterProperty(String key, String value, String delimiter) {
    @Nullable String prior = getFrontMatterProperty(key);
    if (null == prior) setFrontMatterProperty(key, value);
    else setFrontMatterProperty(key, prior + delimiter + value);
  }
  
  /**
   * Gets front matter property.
   *
   * @param key the key
   * @return the front matter property
   */
  @Nullable
  String getFrontMatterProperty(String key);
  
  /**
   * Gets name.
   *
   * @return the name
   */
  String getName();
  
  /**
   * Gets resource dir.
   *
   * @return the resource dir
   */
  @Nonnull
  File getResourceDir();
  
  /**
   * Gets max out size.
   *
   * @return the max out size
   */
  int getMaxOutSize();
  
  /**
   * Sets max out size.
   *
   * @param size the size
   * @return the max out size
   */
  @Nonnull
  NotebookOutput setMaxOutSize(int size);
}
