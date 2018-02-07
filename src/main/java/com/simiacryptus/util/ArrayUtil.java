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

package com.simiacryptus.util;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Array util.
 */
public class ArrayUtil {
  
  /**
   * Add double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static @NotNull double[] add(final @NotNull double[] a, final @NotNull double[] b) {
    return ArrayUtil.op(a, b, (x, y) -> x + y);
  }
  
  /**
   * Add list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> add(final @NotNull List<double[]> a, final @NotNull List<double[]> b) {
    return ArrayUtil.op(a, b, (x, y) -> x + y);
  }
  
  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(final @NotNull double[] a, final @NotNull double[] b) {
    return ArrayUtil.sum(ArrayUtil.op(a, b, (x, y) -> x * y));
  }
  
  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(final @NotNull List<double[]> a, final @NotNull List<double[]> b) {
    return ArrayUtil.sum(ArrayUtil.multiply(a, b));
  }
  
  /**
   * Magnitude double.
   *
   * @param a the a
   * @return the double
   */
  public static double magnitude(final @NotNull double[] a) {
    return Math.sqrt(ArrayUtil.dot(a, a));
  }
  
  /**
   * Mean double.
   *
   * @param op the op
   * @return the double
   */
  public static double mean(final @NotNull double[] op) {
    return ArrayUtil.sum(op) / op.length;
  }
  
  /**
   * Minus list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> minus(final @NotNull List<double[]> a, final @NotNull List<double[]> b) {
    return ArrayUtil.op(a, b, (x, y) -> x - y);
  }
  
  /**
   * Multiply double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static @NotNull double[] multiply(final @NotNull double[] a, final double b) {
    return ArrayUtil.op(a, (x) -> x * b);
  }
  
  /**
   * Multiply double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static @NotNull double[] multiply(final @NotNull double[] a, final @NotNull double[] b) {
    return ArrayUtil.op(a, b, (x, y) -> x * y);
  }
  
  /**
   * Multiply list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static @NotNull List<double[]> multiply(final @NotNull List<double[]> a, final double b) {
    return ArrayUtil.op(a, x -> x * b);
  }
  
  /**
   * Multiply list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> multiply(final @NotNull List<double[]> a, final @NotNull List<double[]> b) {
    return ArrayUtil.op(a, b, (x, y) -> x * y);
  }
  
  /**
   * Op double [ ].
   *
   * @param a  the a
   * @param b  the b
   * @param fn the fn
   * @return the double [ ]
   */
  public static @NotNull double[] op(final @NotNull double[] a, final @NotNull double[] b, final @NotNull DoubleBinaryOperator fn) {
    assert a.length == b.length;
    final @NotNull double[] c = new double[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = fn.applyAsDouble(a[j], b[j]);
    }
    return c;
  }
  
  /**
   * Op double [ ].
   *
   * @param a  the a
   * @param fn the fn
   * @return the double [ ]
   */
  public static @NotNull double[] op(final @NotNull double[] a, final @NotNull DoubleUnaryOperator fn) {
    final @NotNull double[] c = new double[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = fn.applyAsDouble(a[j]);
    }
    return c;
  }
  
  /**
   * Op list.
   *
   * @param a  the a
   * @param fn the fn
   * @return the list
   */
  public static @NotNull List<double[]> op(final @NotNull List<double[]> a, final @NotNull DoubleUnaryOperator fn) {
    final @NotNull ArrayList<double[]> list = new ArrayList<>();
    for (int i = 0; i < a.size(); i++) {
      final @NotNull double[] c = new double[a.get(i).length];
      for (int j = 0; j < a.get(i).length; j++) {
        c[j] = fn.applyAsDouble(a.get(i)[j]);
      }
      list.add(c);
    }
    return list;
  }
  
  /**
   * Op list.
   *
   * @param a  the a
   * @param b  the b
   * @param fn the fn
   * @return the list
   */
  public static List<double[]> op(final @NotNull List<double[]> a, final @NotNull List<double[]> b, final @NotNull DoubleBinaryOperator fn) {
    assert a.size() == b.size();
    return IntStream.range(0, a.size()).parallel().mapToObj(i -> {
      assert a.get(i).length == b.get(i).length;
      final @NotNull double[] c = new double[a.get(i).length];
      for (int j = 0; j < a.get(i).length; j++) {
        c[j] = fn.applyAsDouble(a.get(i)[j], b.get(i)[j]);
      }
      return c;
    }).collect(Collectors.toList());
  }
  
  /**
   * Subtract double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static @NotNull double[] subtract(final @NotNull double[] a, final @NotNull double[] b) {
    return ArrayUtil.op(a, b, (x, y) -> x - y);
  }
  
  /**
   * Sum double.
   *
   * @param op the op
   * @return the double
   */
  public static double sum(final @NotNull double[] op) {
    return Arrays.stream(op).sum();
  }
  
  /**
   * Sum double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static @NotNull double[] sum(final @NotNull double[] a, final double b) {
    return ArrayUtil.op(a, (x) -> x + b);
  }
  
  /**
   * Sum double.
   *
   * @param a the a
   * @return the double
   */
  public static double sum(final @NotNull List<double[]> a) {
    return a.stream().parallel().mapToDouble(x -> Arrays.stream(x).sum()).sum();
  }
  
}
