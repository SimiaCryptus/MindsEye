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

package com.simiacryptus.util;

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
   * Minus list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> minus(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x - y);
  }
  
  /**
   * Add list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> add(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x + y);
  }
  
  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(List<double[]> a, List<double[]> b) {
    return sum(multiply(a, b));
  }
  
  /**
   * Multiply list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> multiply(List<double[]> a, List<double[]> b) {
    return op(a, b, (x, y) -> x * y);
  }
  
  /**
   * Multiply list.
   *
   * @param a the a
   * @param b the b
   * @return the list
   */
  public static List<double[]> multiply(List<double[]> a, double b) {
    return op(a, x -> x * b);
  }
  
  /**
   * Add double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static double[] add(double[] a, double[] b) {
    return op(a, b, (x, y) -> x + y);
  }
  
  /**
   * Subtract double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static double[] subtract(double[] a, double[] b) {
    return op(a, b, (x, y) -> x - y);
  }
  
  /**
   * Multiply double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static double[] multiply(double[] a, double[] b) {
    return op(a, b, (x, y) -> x * y);
  }
  
  /**
   * Dot double.
   *
   * @param a the a
   * @param b the b
   * @return the double
   */
  public static double dot(double[] a, double[] b) {
    return sum(op(a, b, (x, y) -> x * y));
  }
  
  /**
   * Magnitude double.
   *
   * @param a the a
   * @return the double
   */
  public static double magnitude(double[] a) {
    return Math.sqrt(dot(a, a));
  }
  
  /**
   * Sum double.
   *
   * @param op the op
   * @return the double
   */
  public static double sum(double[] op) {
    return Arrays.stream(op).sum();
  }
  
  /**
   * Mean double.
   *
   * @param op the op
   * @return the double
   */
  public static double mean(double[] op) {
    return sum(op) / op.length;
  }
  
  /**
   * Multiply double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static double[] multiply(double[] a, double b) {
    return op(a, (x) -> x * b);
  }
  
  /**
   * Sum double [ ].
   *
   * @param a the a
   * @param b the b
   * @return the double [ ]
   */
  public static double[] sum(double[] a, double b) {
    return op(a, (x) -> x + b);
  }
  
  /**
   * Sum double.
   *
   * @param a the a
   * @return the double
   */
  public static double sum(List<double[]> a) {
    return a.stream().parallel().mapToDouble(x -> Arrays.stream(x).sum()).sum();
  }
  
  /**
   * Op list.
   *
   * @param a  the a
   * @param b  the b
   * @param fn the fn
   * @return the list
   */
  public static List<double[]> op(List<double[]> a, List<double[]> b, DoubleBinaryOperator fn) {
    assert (a.size() == b.size());
    return IntStream.range(0, a.size()).parallel().mapToObj(i -> {
      assert (a.get(i).length == b.get(i).length);
      double[] c = new double[a.get(i).length];
      for (int j = 0; j < a.get(i).length; j++) {
        c[j] = fn.applyAsDouble(a.get(i)[j], b.get(i)[j]);
      }
      return c;
    }).collect(Collectors.toList());
  }
  
  /**
   * Op double [ ].
   *
   * @param a  the a
   * @param b  the b
   * @param fn the fn
   * @return the double [ ]
   */
  public static double[] op(double[] a, double[] b, DoubleBinaryOperator fn) {
    assert (a.length == b.length);
    double[] c = new double[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = fn.applyAsDouble(a[j], b[j]);
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
  public static List<double[]> op(List<double[]> a, DoubleUnaryOperator fn) {
    ArrayList<double[]> list = new ArrayList<>();
    for (int i = 0; i < a.size(); i++) {
      double[] c = new double[a.get(i).length];
      for (int j = 0; j < a.get(i).length; j++) {
        c[j] = fn.applyAsDouble(a.get(i)[j]);
      }
      list.add(c);
    }
    return list;
  }
  
  /**
   * Op double [ ].
   *
   * @param a  the a
   * @param fn the fn
   * @return the double [ ]
   */
  public static double[] op(double[] a, DoubleUnaryOperator fn) {
    double[] c = new double[a.length];
    for (int j = 0; j < a.length; j++) {
      c[j] = fn.applyAsDouble(a[j]);
    }
    return c;
  }
  
}
