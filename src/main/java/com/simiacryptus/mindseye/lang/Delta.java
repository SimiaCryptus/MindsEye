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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * An arithmetic evalInputDelta being staged to effect an in-memory change to a double[] array. In comparison apply the State
 * class via geometric analogy, this would be a vector whereas State is a point.
 *
 * @param <K> the type parameter
 */
public class Delta<K> extends DoubleBuffer<K> {
  /**
   * The Delta compensation.
   */
  @Nullable
  protected double[] deltaCompensation;

  /**
   * Instantiates a new Delta.
   *
   * @param layer  the key
   * @param target the target
   */
  public Delta(@Nonnull final K layer, @Nullable final double[] target) {
    this(layer, target, null == target ? null : RecycleBin.DOUBLES.obtain(target.length));
  }

  /**
   * Instantiates a new Delta.
   *
   * @param layer  the key
   * @param target the target
   * @param delta  the evalInputDelta
   */
  public Delta(@Nonnull final K layer, final double[] target, @Nonnull final double[] delta) {
    this(layer, target, delta, RecycleBin.DOUBLES.obtain(delta.length));
  }

  /**
   * Instantiates a new Delta.
   *
   * @param layer             the key
   * @param target            the target
   * @param delta             the doubles
   * @param deltaCompensation the evalInputDelta compensation
   */
  protected Delta(@Nonnull final K layer, @Nullable final double[] target, @Nullable final double[] delta, final double[] deltaCompensation) {
    super(layer, target, delta);
    if (null == target) throw new IllegalArgumentException();
    assert null == delta || target.length == delta.length;
    //if(null == array) throw new IllegalArgumentException();
    this.deltaCompensation = deltaCompensation;
  }

  /**
   * Accumulate.
   *
   * @param data             the data
   * @param delta            the evalInputDelta
   * @param dataCompensation the data compensation
   */
  public static void accumulate(@Nonnull final double[] data, final double[] delta, @Nullable final double[] dataCompensation) {
    synchronized (data) {
      for (int i = 0; i < data.length; i++) {
        final double sum = data[i];
        final double input = delta[i];
        double c = null == dataCompensation ? 0 : dataCompensation[i];
        if (Math.abs(sum) >= Math.abs(input)) {
          final double y = sum - c;
          final double t = input + y;
          c = t - input - y;
          data[i] = t;
          if (null != dataCompensation) {
            dataCompensation[i] = c;
          }
        } else {
          final double y = input - c;
          final double t = sum + y;
          c = t - sum - y;
          data[i] = t;
          if (null != dataCompensation) {
            dataCompensation[i] = c;
          }
        }
        if (!Double.isFinite(data[i])) data[i] = 0;
      }
    }
  }

  /**
   * Accumulate.
   *
   * @param factor the factor
   */
  public final void accumulate(final double factor) {
    synchronized (target) {
      assert Arrays.stream(target).allMatch(Double::isFinite);
      @Nullable final double[] delta = getDelta();
      for (int i = 0; i < length(); i++) {
        target[i] += delta[i] * factor;
        if (!Double.isFinite(target[i])) target[i] = 0;
      }
      assert Arrays.stream(target).allMatch(Double::isFinite);
    }
  }

  /**
   * Add in place evalInputDelta.
   *
   * @param buffer the buffer
   * @return the evalInputDelta
   */
  @Nonnull
  public Delta<K> addInPlace(@Nonnull final Delta<K> buffer) {
    assertAlive();
    return addInPlace(buffer.delta).addInPlace(buffer.deltaCompensation);
  }

  /**
   * Accumulate evalInputDelta.
   *
   * @param data the data
   * @return the evalInputDelta
   */
  @Nonnull
  public Delta<K> addInPlace(@Nonnull final double[] data) {
    assert data.length == this.target.length;
    //assert Arrays.stream(data).allMatch(Double::isFinite);
    Delta.accumulate(getDelta(), data, deltaCompensation);
    //assert Arrays.stream(read()).allMatch(Double::isFinite);
    return this;
  }


  @Nonnull
  @Override
  public Delta<K> copy() {
    assertAlive();
    return new Delta<K>(key, target, RecycleBin.DOUBLES.copyOf(delta, length()), RecycleBin.DOUBLES.copyOf(deltaCompensation, length()));
  }

  @Override
  protected void _free() {
    super._free();
    if (null != deltaCompensation) {
      if (RecycleBin.DOUBLES.want(deltaCompensation.length)) {
        RecycleBin.DOUBLES.recycle(deltaCompensation, deltaCompensation.length);
      }
      deltaCompensation = null;
    }
  }

  @Nonnull
  @Override
  public Delta<K> map(@Nonnull final DoubleUnaryOperator mapper) {
    return new Delta<K>(key, target, Arrays.stream(getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }

  /**
   * Scale evalInputDelta.
   *
   * @param f the f
   * @return the evalInputDelta
   */
  @Nonnull
  public Delta<K> scale(final double f) {
    return map(x -> x * f);
  }

  @Nonnull
  @Override
  public Delta<K> set(final double[] data) {
    super.set(data);
    return this;
  }
}
