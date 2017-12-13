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

package com.simiacryptus.util.lang;

/**
 * The type Resource pool.
 *
 * @param <T> the type parameter
 */
public abstract class ResourcePool<T> {
  
  private final ThreadLocal<T> currentValue = new ThreadLocal<>();
  private final java.util.HashSet<T> all;
  private final java.util.concurrent.LinkedBlockingQueue<T> pool = new java.util.concurrent.LinkedBlockingQueue<>();
  private final int maxItems;
  
  /**
   * Instantiates a new Resource pool.
   *
   * @param maxItems the max items
   */
  public ResourcePool(int maxItems) {
    super();
    this.maxItems = maxItems;
    this.all = new java.util.HashSet<>(this.maxItems);
  }
  
  /**
   * Create t.
   *
   * @return the t
   */
  public abstract T create();
  
  /**
   * With.
   *
   * @param f the f
   */
  public void with(final java.util.function.Consumer<T> f) {
    T prior = currentValue.get();
    if (null != prior) {
      f.accept(prior);
    }
    else {
      T poll = get();
      try {
        currentValue.set(poll);
        f.accept(poll);
      } finally {
        this.pool.add(poll);
        currentValue.remove();
      }
    }
  }
  
  /**
   * Get t.
   *
   * @return the t
   */
  public T get() {
    T poll = this.pool.poll();
    if (null == poll) {
      synchronized (this.all) {
        if (this.all.size() < this.maxItems) {
          poll = create();
          this.all.add(poll);
        }
      }
    }
    if (null == poll) {
      try {
        poll = this.pool.take();
      } catch (final InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
    return poll;
  }
  
  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return all.size();
  }
}
