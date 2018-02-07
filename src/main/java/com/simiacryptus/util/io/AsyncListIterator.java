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

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Iterator;
import java.util.List;

/**
 * Created by Andrew Charneski on 3/11/2017.
 *
 * @param <T> the type parameter
 */
public class AsyncListIterator<T> implements Iterator<T> {
  private final List<T> queue;
  private final Thread thread;
  /**
   * The Index.
   */
  int index = -1;
  
  /**
   * Instantiates a new Async list iterator.
   *
   * @param queue  the queue
   * @param thread the thread
   */
  public AsyncListIterator(final List<T> queue, final Thread thread) {
    this.thread = thread;
    this.queue = queue;
  }
  
  @Override
  protected void finalize() throws Throwable {
    super.finalize();
  }
  
  @Override
  public boolean hasNext() {
    return index < queue.size() || thread.isAlive();
  }
  
  @Override
  public @Nullable T next() {
    try {
      while (hasNext()) {
        if (++index < queue.size()) {
          return queue.get(index);
        }
        else {
          Thread.sleep(100);
        }
      }
      return null;
    } catch (final @NotNull InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
}
