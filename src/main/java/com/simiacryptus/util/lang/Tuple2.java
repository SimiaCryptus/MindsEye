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
 * The type Tuple 2.
 *
 * @param <A> the type parameter
 * @param <B> the type parameter
 */
public class Tuple2<A, B> {
  /**
   * The 1.
   */
  public final A _1;
  /**
   * The 2.
   */
  public final B _2;
  
  /**
   * Instantiates a new Tuple 2.
   *
   * @param a the a
   * @param b the b
   */
  public Tuple2(final A a, final B b) {
    _1 = a;
    _2 = b;
  }
  
  /**
   * Gets first.
   *
   * @return the first
   */
  public A getFirst() {
    return _1;
  }
  
  /**
   * Gets second.
   *
   * @return the second
   */
  public B getSecond() {
    return _2;
  }
}
