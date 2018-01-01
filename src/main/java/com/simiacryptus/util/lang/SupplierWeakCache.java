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

import java.lang.ref.WeakReference;
import java.util.function.Supplier;

/**
 * The type Supplier weak cache.
 *
 * @param <T> the type parameter
 */
public class SupplierWeakCache<T> implements Supplier<T> {
  private final Supplier<T> fn;
  private WeakReference<T> ptr;
  
  /**
   * Instantiates a new Supplier weak cache.
   *
   * @param fn the fn
   */
  public SupplierWeakCache(final Supplier<T> fn) {
    this.fn = fn;
    this.ptr = null;
  }
  
  @Override
  public T get() {
    T x = null == ptr ? null : ptr.get();
    if (null == x) {
      x = fn.get();
      ptr = new WeakReference<>(x);
    }
    return x;
  }
}
