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

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * An arithmetic delta being staged to effect an in-memory change to a double[] array.
 * In comparison with the State class via geometric analogy, this would be a vector whereas State is a point.
 *
 * @param <K> the type parameter
 */
public class Delta<K> extends DoubleBuffer<K> {
  /**
   * The Delta compensation.
   */
  protected double[] deltaCompensation;
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer  the layer
   * @param target the target
   */
  public Delta(final K layer, final double[] target) {
    this(layer, target, null == target ? null : RecycleBin.DOUBLES.obtain(target.length));
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer  the layer
   * @param target the target
   * @param delta  the delta
   */
  public Delta(final K layer, final double[] target, final double[] delta) {
    this(layer, target, delta, RecycleBin.DOUBLES.obtain(delta.length));
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer             the layer
   * @param target            the target
   * @param doubles           the doubles
   * @param deltaCompensation the delta compensation
   */
  protected Delta(final K layer, final double[] target, final double[] doubles, final double[] deltaCompensation) {
    super(layer, target, doubles);
    if (null == target) throw new IllegalArgumentException();
    //if(null == array) throw new IllegalArgumentException();
    this.deltaCompensation = deltaCompensation;
  }
  
  /**
   * Accumulate.
   *
   * @param data             the data
   * @param delta            the delta
   * @param dataCompensation the data compensation
   */
  public static void accumulate(final double[] data, final double[] delta, final double[] dataCompensation) {
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
        }
        else {
          final double y = input - c;
          final double t = sum + y;
          c = t - sum - y;
          data[i] = t;
          if (null != dataCompensation) {
            dataCompensation[i] = c;
          }
        }
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
      final double[] delta = getDelta();
      for (int i = 0; i < length(); i++) {
        target[i] += delta[i] * factor;
      }
      assert Arrays.stream(target).allMatch(Double::isFinite);
    }
  }
  
  /**
   * Add in place delta.
   *
   * @param buffer the buffer
   * @return the delta
   */
  public Delta<K> addInPlace(final Delta<K> buffer) {
    return addInPlace(buffer.delta).addInPlace(buffer.deltaCompensation);
  }
  
  /**
   * Accumulate delta.
   *
   * @param data the data
   * @return the delta
   */
  public Delta<K> addInPlace(final double[] data) {
    //assert Arrays.stream(data).allMatch(Double::isFinite);
    Delta.accumulate(getDelta(), data, deltaCompensation);
    //assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
  }
  
  
  @Override
  public Delta<K> copy() {
    return new Delta<K>(layer, target, RecycleBin.DOUBLES.copyOf(delta), RecycleBin.DOUBLES.copyOf(deltaCompensation));
  }
  
  @Override
  protected void finalize() throws Throwable {
    if (null != delta) {
      RecycleBin.DOUBLES.recycle(delta);
      delta = null;
    }
    if (null != deltaCompensation) {
      RecycleBin.DOUBLES.recycle(deltaCompensation);
      deltaCompensation = null;
    }
    super.finalize();
  }
  
  @Override
  public Delta<K> map(final DoubleUnaryOperator mapper) {
    return new Delta<K>(layer, target, Arrays.stream(getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }
  
  /**
   * Scale delta.
   *
   * @param f the f
   * @return the delta
   */
  public Delta<K> scale(final double f) {
    return map(x -> x * f);
  }
  
  @Override
  public Delta<K> set(final double[] data) {
    super.set(data);
    return this;
  }
}
