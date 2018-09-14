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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * A generic alternate memory buffer being staged in relation to an existing double[] array.
 *
 * @param <K> the type parameter
 */
public class DoubleBuffer<K extends ReferenceCounting> extends ReferenceCountingBase {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DoubleBuffer.class);
  /**
   * The LayerBase.
   */
  @Nonnull
  public final K layer;
  /**
   * The Target.
   */
  public final double[] target;
  /**
   * The Delta.
   */
  @Nullable
  protected volatile double[] delta;
  
  /**
   * Instantiates a new Double buffer.
   *
   * @param layer  the layer
   * @param target the target
   */
  public DoubleBuffer(@Nonnull final K layer, final double[] target) {
    this.layer = layer;
    layer.addRef(this);
    this.target = target;
    this.delta = null;
  }
  
  /**
   * Instantiates a new Double buffer.
   *
   * @param layer  the layer
   * @param target the target
   * @param delta  the evalInputDelta
   */
  public DoubleBuffer(@Nonnull final K layer, final double[] target, final double[] delta) {
    this.layer = layer;
    layer.addRef(this);
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
  public static boolean areEqual(@Nonnull final double[] l, @Nonnull final double[] r) {
    if (r.length != l.length) throw new IllegalArgumentException();
    for (int i = 0; i < r.length; i++) {
      if (r[i] != l[i]) return false;
    }
    return true;
  }
  
  /**
   * Copy evalInputDelta.
   *
   * @return the evalInputDelta
   */
  @Nullable
  public DoubleBuffer<K> copy() {
    assertAlive();
    return new DoubleBuffer<K>(layer, target, RecycleBin.DOUBLES.copyOf(delta, length()));
  }
  
  /**
   * Delta statistics double array stats facade.
   *
   * @return the double array stats facade
   */
  @Nonnull
  public DoubleArrayStatsFacade deltaStatistics() {
    return new DoubleArrayStatsFacade(getDelta());
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(@Nonnull final DoubleBuffer<K> right) {
    if (this.target != right.target) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same buffer. %s != %s", this.layer, right.layer));
    }
    if (!this.layer.equals(right.layer)) {
      throw new IllegalArgumentException(String.format("Deltas are not based on same layer. %s != %s", this.layer, right.layer));
    }
    @Nullable final double[] l = this.getDelta();
    @Nullable final double[] r = right.getDelta();
    assert l.length == r.length;
    final double[] array = IntStream.range(0, l.length).mapToDouble(i -> l[i] * r[i]).toArray();
    return Arrays.stream(array).summaryStatistics().getSum();
  }
  
  @Nullable
  public double[] getDeltaAndFree() {
    double[] delta = getDelta();
    freeRef();
    return delta;
  }
  
  /**
   * Get evalInputDelta double [ ].
   *
   * @return the double [ ]
   */
  @Nullable
  public double[] getDelta() {
    assertAlive();
    if (null == delta) {
      synchronized (this) {
        if (null == delta) {
          delta = RecycleBin.DOUBLES.obtain(target.length);
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
  public CharSequence getId() {
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
   * Map evalInputDelta.
   *
   * @param mapper the mapper
   * @return the evalInputDelta
   */
  @Nonnull
  public DoubleBuffer<K> map(@Nonnull final DoubleUnaryOperator mapper) {
    return new DoubleBuffer<K>(this.layer, this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }
  
  /**
   * Set evalInputDelta.
   *
   * @param data the data
   * @return the evalInputDelta
   */
  @Nonnull
  public DoubleBuffer<K> set(@Nonnull final double[] data) {
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
  @Nonnull
  public DoubleArrayStatsFacade targetStatistics() {
    return new DoubleArrayStatsFacade(target);
  }
  
  @Nonnull
  @Override
  public String toString() {
    @Nonnull final StringBuilder builder = new StringBuilder();
    builder.append(getClass().getSimpleName());
    builder.append("/");
    builder.append(this.layer);
    return builder.toString();
  }
  
  @Override
  protected void _free() {
    layer.freeRef();
    @Nullable double[] delta = this.delta;
    if (null != delta) {
      if (RecycleBin.DOUBLES.want(delta.length)) {
        RecycleBin.DOUBLES.recycle(this.delta, delta.length);
      }
      this.delta = null;
    }
  }
}
