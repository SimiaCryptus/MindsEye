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

package com.simiacryptus.mindseye.lang;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * The type Delta.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class Delta {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(Delta.class);
  /**
   * The Layer.
   */
  public final NNLayer layer;
  /**
   * The Target.
   */
  public final double[] target;
  /**
   * The Delta.
   */
  protected double[] delta;
  
  /**
   * Instantiates a new Delta.
   *
   * @param target the target
   * @param delta  the delta
   * @param layer  the layer
   */
  public Delta(final double[] target, final double[] delta, final NNLayer layer) {
    if (null == target) throw new IllegalArgumentException();
    //if(null == array) throw new IllegalArgumentException();
    this.target = target;
    this.layer = layer;
    this.delta = delta;
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param values the values
   * @param layer  the layer
   */
  public Delta(final double[] values, final NNLayer layer) {
    if (null == values) throw new IllegalArgumentException();
    this.target = values;
    this.layer = layer;
    this.delta = new double[values.length];
    Arrays.fill(this.getDelta(), 0);
  }
  
  /**
   * Accumulate.
   *
   * @param data  the data
   * @param delta the delta
   */
  public static void accumulate(double[] data, double[] delta) {
    Arrays.parallelSetAll(data, i -> data[i] + delta[i]);
  }
  
  /**
   * Are equal boolean.
   *
   * @param l the l
   * @param r the r
   * @return the boolean
   */
  public static boolean areEqual(double[] l, double[] r) {
    assert (r.length == l.length);
    for (int i = 0; i < r.length; i++) {
      if (r[i] != l[i]) return false;
    }
    return true;
  }
  
  /**
   * Accumulate delta.
   *
   * @param data the data
   * @return the delta
   */
  public Delta accumulate(final double[] data) {
    //assert Arrays.stream(data).allMatch(Double::isFinite);
    double[] delta = this.getDelta();
    accumulate(delta, data);
    //assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
  }
  
  /**
   * Set delta.
   *
   * @param data the data
   * @return the delta
   */
  public Delta set(final double[] data) {
    assert Arrays.stream(data).allMatch(Double::isFinite);
    Arrays.parallelSetAll(this.getDelta(), i -> data[i]);
    assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
  }
  
  /**
   * Copy delta double [ ].
   *
   * @return the double [ ]
   */
  public double[] copyDelta() {
    return null == getDelta() ? null : Arrays.copyOf(getDelta(), getDelta().length);
  }
  
  /**
   * Gets id.
   *
   * @return the id
   */
  public String getId() {
    return this.layer.getId();
  }
  
  /**
   * Is frozen boolean.
   *
   * @return the boolean
   */
  public boolean isFrozen() {
    return false;
  }
  
  /**
   * Length int.
   *
   * @return the int
   */
  public int length() {
    return this.target.length;
  }
  
  /**
   * Map delta.
   *
   * @param mapper the mapper
   * @return the delta
   */
  public Delta map(final DoubleUnaryOperator mapper) {
    return new Delta(this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
  }
  
  /**
   * Scale delta.
   *
   * @param f the f
   * @return the delta
   */
  public Delta scale(final double f) {
    return map(x -> x * f);
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer.getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer.getId());
    return builder.toString();
  }
  
  /**
   * Accumulate.
   *
   * @param factor the factor
   */
  public final synchronized void accumulate(final double factor) {
    assert Arrays.stream(target).allMatch(Double::isFinite);
    double[] calcVector = this.getDelta();
    if (null == calcVector) {
      return;
    }
    calcVector = Arrays.copyOf(calcVector, calcVector.length);
    for (int i = 0; i < this.getDelta().length; i++) {
      calcVector[i] = calcVector[i] * factor;
    }
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.target[i] = this.target[i] + calcVector[i];
    }
    assert Arrays.stream(target).allMatch(Double::isFinite);
  }
  
  /**
   * Overwrite.
   */
  public final synchronized void overwrite() {
    final int dim = length();
    for (int i = 0; i < dim; i++) {
      this.target[i] = this.getDelta()[i];
    }
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(Delta right) {
    if (this.target != right.target) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same buffer. %s != %s", this.layer, right.layer));
    }
    if (!this.layer.equals(right.layer)) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same layer. %s != %s", this.layer, right.layer));
    }
    assert (this.getDelta().length == right.getDelta().length);
    return IntStream.range(0, this.getDelta().length).mapToDouble(i -> getDelta()[i] * right.getDelta()[i]).sum();
  }
  
  /**
   * Sum double.
   *
   * @return the double
   */
  public double sum() {
    return Arrays.stream(this.getDelta()).sum();
  }
  
  /**
   * Sum sq double.
   *
   * @return the double
   */
  public double sumSq() {
    return Arrays.stream(this.getDelta()).map(x -> x * x).sum();
  }
  
  /**
   * Copy delta.
   *
   * @return the delta
   */
  public Delta copy() {
    return new Delta(target, copyDelta(), layer);
  }
  
  /**
   * Get delta double [ ].
   *
   * @return the double [ ]
   */
  public double[] getDelta() {
    return delta;
  }
  
  /**
   * Are equal boolean.
   *
   * @return the boolean
   */
  public boolean areEqual() {
    return areEqual(getDelta(), target);
  }
  
}
