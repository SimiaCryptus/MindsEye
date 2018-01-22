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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * A generic alternate memory buffer being staged in relation to an existing double[] array.
 *
 * @param <K> the type parameter
 */
public class DoubleBuffer<K> {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DoubleBuffer.class);
  /**
   * The Layer.
   */
  public final K layer;
  /**
   * The Target.
   */
  public final double[] target;
  /**
   * The Delta.
   */
  protected volatile double[] delta;
  
  /**
   * Instantiates a new Double buffer.
   *
   * @param layer  the layer
   * @param target the target
   */
  public DoubleBuffer(final K layer, final double[] target) {
    this.layer = layer;
    this.target = target;
    this.delta = null;
  }
  
  /**
   * Instantiates a new Double buffer.
   *
   * @param layer  the layer
   * @param target the target
   * @param delta  the delta
   */
  public DoubleBuffer(final K layer, final double[] target, final double[] delta) {
    this.layer = layer;
    this.target = target;
    this.delta = delta;
  }
  
  /**
   * Are equal boolean.
   *
   * @param l the l
   * @param r the r
   * @return the boolean
   */
  public static boolean areEqual(final double[] l, final double[] r) {
    if (r.length != l.length) throw new IllegalArgumentException();
    for (int i = 0; i < r.length; i++) {
      if (r[i] != l[i]) return false;
    }
    return true;
  }
  
  /**
   * Copy delta.
   *
   * @return the delta
   */
  public DoubleBuffer<K> copy() {
    return new DoubleBuffer<K>(layer, target, RecycleBinLong.DOUBLES.copyOf(delta, length()));
  }
  
  /**
   * Delta statistics double array stats facade.
   *
   * @return the double array stats facade
   */
  public DoubleArrayStatsFacade deltaStatistics() {
    return new DoubleArrayStatsFacade(getDelta());
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(final DoubleBuffer<K> right) {
    if (this.target != right.target) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same buffer. %s != %s", this.layer, right.layer));
    }
    if (!this.layer.equals(right.layer)) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same layer. %s != %s", this.layer, right.layer));
    }
    final double[] l = this.getDelta();
    final double[] r = right.getDelta();
    assert l.length == r.length;
    final double[] array = IntStream.range(0, l.length).mapToDouble(i -> l[i] * r[i]).toArray();
    return Arrays.stream(array).summaryStatistics().getSum();
  }
  
  /**
   * Get delta double [ ].
   *
   * @return the double [ ]
   */
  public double[] getDelta() {
    if (null == delta) {
      synchronized (this) {
        if (null == delta) {
          delta = RecycleBinLong.DOUBLES.obtain(target.length);
        }
      }
    }
    return delta;
  }
  
  /**
   * Gets id.
   *
   * @return the id
   */
  public String getId() {
    return this.layer.toString();
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
  public DoubleBuffer<K> map(final DoubleUnaryOperator mapper) {
    return new DoubleBuffer<K>(this.layer, this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }
  
  /**
   * Set delta.
   *
   * @param data the data
   * @return the delta
   */
  public DoubleBuffer<K> set(final double[] data) {
    assert Arrays.stream(data).allMatch(Double::isFinite);
    Arrays.parallelSetAll(this.getDelta(), i -> data[i]);
    assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
  }
  
  /**
   * Target statistics double array stats facade.
   *
   * @return the double array stats facade
   */
  public DoubleArrayStatsFacade targetStatistics() {
    return new DoubleArrayStatsFacade(target);
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer);
    return builder.toString();
  }
  
}
