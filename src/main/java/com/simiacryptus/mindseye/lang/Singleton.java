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

package com.simiacryptus.mindseye.lang;

import java.util.concurrent.BlockingDeque;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.function.Supplier;

/**
 * The type Singleton.
 *
 * @param <T> the type parameter
 */
public class Singleton<T> implements Supplier<T> {
  private final BlockingDeque<T> deque = new LinkedBlockingDeque<>();
  
  /**
   * Instantiates a new Singleton.
   */
  public Singleton() {}
  
  @Override
  public T get() {
    try {
      T take = deque.take();
      deque.add(take);
      return take;
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Set.
   *
   * @param obj the obj
   * @return the singleton
   */
  public Singleton<T> set(T obj) {
    assert deque.isEmpty();
    deque.add(obj);
    return this;
  }
}
