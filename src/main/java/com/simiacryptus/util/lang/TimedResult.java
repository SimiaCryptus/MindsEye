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
 * The type Timed result.
 *
 * @param <T> the type parameter
 */
public class TimedResult<T> {
  /**
   * The Obj.
   */
  public final T result;
  /**
   * The Time nanos.
   */
  public final long timeNanos;
  
  /**
   * Instantiates a new Timed result.
   *
   * @param result    the obj
   * @param timeNanos the time nanos
   */
  public TimedResult(T result, long timeNanos) {
    this.result = result;
    this.timeNanos = timeNanos;
  }
  
  /**
   * Time timed result.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the timed result
   */
  public static <T> TimedResult<T> time(UncheckedSupplier<T> fn) {
    long start = System.nanoTime();
    T result = null;
    try {
      result = fn.get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return new TimedResult(result, System.nanoTime() - start);
  }
  
  /**
   * Seconds double.
   *
   * @return the double
   */
  public double seconds() {
    return timeNanos / 1000000000.0;
  }
}
