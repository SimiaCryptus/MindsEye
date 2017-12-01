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
 * A generic alternate memory buffer being staged in relation to an existing double[] array.
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
  protected double[] delta;
  
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
   * Instantiates a new Double buffer.
   *
   * @param target the target
   * @param delta  the delta
   * @param layer  the layer
   */
  public DoubleBuffer(double[] target, double[] delta, K layer) {
    this.layer = layer;
    this.target = target;
    this.delta = delta;
  }
  
  /**
   * Instantiates a new Double buffer.
   *
   * @param target the target
   * @param layer  the layer
   */
  public DoubleBuffer(double[] target, K layer) {
    this.layer = layer;
    this.target = target;
    this.delta = null;
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
   * Delta statistics double array stats facade.
   *
   * @return the double array stats facade
   */
  public DoubleArrayStatsFacade deltaStatistics() {
    return new DoubleArrayStatsFacade(delta);
  }
  
  /**
   * Target statistics double array stats facade.
   *
   * @return the double array stats facade
   */
  public DoubleArrayStatsFacade targetStatistics() {
    return new DoubleArrayStatsFacade(target);
  }
  
  /**
   * Set delta.
   *
   * @param data the data
   * @return the delta
   */
  public DoubleBuffer set(final double[] data) {
    assert Arrays.stream(data).allMatch(Double::isFinite);
    Arrays.parallelSetAll(this.getDelta(), i -> data[i]);
    assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
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
  public DoubleBuffer map(final DoubleUnaryOperator mapper) {
    return new DoubleBuffer(this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer);
    return builder.toString();
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(DoubleBuffer right) {
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
   * Copy delta.
   *
   * @return the delta
   */
  public DoubleBuffer copy() {
    return new DoubleBuffer(target, DoubleArrays.copyOf(delta), layer);
  }
  
  /**
   * Get delta double [ ].
   *
   * @return the double [ ]
   */
  public double[] getDelta() {
    if(null == delta) {
      synchronized (this) {
        if(null == delta) {
          delta = DoubleArrays.obtain(target.length);
        }
      }
    }
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
  
  /**
   * The type Double array stats facade.
   */
  public static class DoubleArrayStatsFacade {
    /**
     * The Data.
     */
    public final double[] data;
  
    /**
     * Instantiates a new Double array stats facade.
     *
     * @param data the data
     */
    public DoubleArrayStatsFacade(double[] data) {
      this.data = data;
    }
  
  
    /**
     * Sum double.
     *
     * @return the double
     */
    public double sum() {
      return Arrays.stream(data).sum();
    }
  
    /**
     * Sum sq double.
     *
     * @return the double
     */
    public double sumSq() {
      return Arrays.stream(data).map(x -> x * x).sum();
    }
  
    /**
     * Rms double.
     *
     * @return the double
     */
    public double rms() {
      return Math.sqrt(sumSq() / length());
    }
  
    /**
     * Length int.
     *
     * @return the int
     */
    public int length() {
      return data.length;
    }
  }
}
