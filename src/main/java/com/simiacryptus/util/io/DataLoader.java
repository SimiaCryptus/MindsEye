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

import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;


/**
 * The type Data loader.
 *
 * @param <T> the type parameter
 */
public abstract class DataLoader<T> {
  private final List<T> queue = Collections.synchronizedList(new ArrayList<>());
  private volatile @Nullable Thread thread;
  
  /**
   * Clear.
   *
   * @throws InterruptedException the interrupted exception
   */
  public void clear() throws InterruptedException {
    if (thread != null) {
      synchronized (this) {
        if (thread != null) {
          thread.interrupt();
          thread.join();
          thread = null;
          queue.clear();
        }
      }
    }
  }
  
  /**
   * Read.
   *
   * @param queue the queue
   */
  protected abstract void read(List<T> queue);
  
  /**
   * Stop.
   */
  public void stop() {
    if (thread != null) {
      thread.interrupt();
    }
    try {
      thread.join();
    } catch (final @NotNull InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }
  
  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<T> stream() {
    if (thread == null) {
      synchronized (this) {
        if (thread == null) {
          thread = new Thread(() -> read(queue));
          thread.setDaemon(true);
          thread.start();
        }
      }
    }
    final @Nullable Iterator<T> iterator = new AsyncListIterator<>(queue, thread);
    return StreamSupport.stream(Spliterators.spliteratorUnknownSize(iterator, Spliterator.DISTINCT), false).filter(x -> x != null);
  }
}
