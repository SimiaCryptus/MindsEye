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

package com.simiacryptus.util.lang;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The type Static resource pool.
 *
 * @param <T> the type parameter
 */
public class StaticResourcePool<T> {
  
  @javax.annotation.Nonnull
  private final List<T> all;
  private final java.util.concurrent.LinkedBlockingQueue<T> pool = new java.util.concurrent.LinkedBlockingQueue<>();
  
  /**
   * Instantiates a new Static resource pool.
   *
   * @param items the items
   */
  public StaticResourcePool(@javax.annotation.Nonnull final List<T> items) {
    super();
    this.all = Collections.unmodifiableList(new ArrayList<>(items));
    pool.addAll(getAll());
  }
  
  /**
   * With u.
   *
   * @param f the f
   */
  public void apply(@javax.annotation.Nonnull final Consumer<T> f) {
    T poll = this.pool.poll();
    if (null == poll) {
      try {
        poll = this.pool.take();
      } catch (@javax.annotation.Nonnull final InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
    try {
      f.accept(poll);
    } finally {
      this.pool.add(poll);
    }
  }
  
  /**
   * Take t.
   *
   * @return the t
   */
  public T take() {
    try {
      return this.pool.take();
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets all.
   *
   * @return the all
   */
  @javax.annotation.Nonnull
  public List<T> getAll() {
    return all;
  }
  
  /**
   * With u.
   *
   * @param <U> the type parameter
   * @param f   the f
   * @return the u
   */
  public <U> U run(@javax.annotation.Nonnull final Function<T, U> f) {
    if (all.isEmpty()) throw new IllegalStateException();
    T poll = this.pool.poll();
    if (null == poll) {
      try {
        poll = this.pool.take();
      } catch (@javax.annotation.Nonnull final InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
    try {
      return f.apply(poll);
    } finally {
      this.pool.add(poll);
    }
  }
  
  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return getAll().size();
  }
}
