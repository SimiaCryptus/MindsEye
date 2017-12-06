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
   * @param delta  the delta
   */
  public Delta(final K layer, final double[] target, final double[] delta) {
    this(layer, target, delta, DoubleArrays.obtain(delta.length));
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer  the layer
   * @param target the target
   */
  public Delta(final K layer, final double[] target) {
    this(layer, target, null == target ? null : DoubleArrays.obtain(target.length));
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer             the layer
   * @param target            the target
   * @param doubles           the doubles
   * @param deltaCompensation the delta compensation
   */
  protected Delta(K layer, double[] target, double[] doubles, double[] deltaCompensation) {
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
  public static void accumulate(double[] data, double[] delta, double[] dataCompensation) {
    for (int i = 0; i < data.length; i++) {
      double sum = data[i];
      double input = delta[i];
      double c = null == dataCompensation ? 0 : dataCompensation[i];
      if (Math.abs(sum) >= Math.abs(input)) {
        double y = sum - c;
        double t = input + y;
        c = (t - input) - y;
        data[i] = t;
        if (null != dataCompensation) dataCompensation[i] = c;
      }
      else {
        double y = input - c;
        double t = sum + y;
        c = (t - sum) - y;
        data[i] = t;
        if (null != dataCompensation) dataCompensation[i] = c;
      }
    }
  }
  
  /**
   * Accumulate delta.
   *
   * @param data the data
   * @return the delta
   */
  public Delta<K> addInPlace(final double[] data) {
    //assert Arrays.stream(data).allMatch(Double::isFinite);
    accumulate(getDelta(), data, deltaCompensation);
    //assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
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
  
  public Delta<K> map(final DoubleUnaryOperator mapper) {
    return new Delta(this.layer, this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }
  
  
  /**
   * Accumulate.
   *
   * @param factor the factor
   */
  public final synchronized void accumulate(final double factor) {
    assert Arrays.stream(target).allMatch(Double::isFinite);
    double[] delta = this.getDelta();
    for (int i = 0; i < length(); i++) {
      target[i] += delta[i] * factor;
    }
    assert Arrays.stream(target).allMatch(Double::isFinite);
  }
  
  @Override
  protected void finalize() throws Throwable {
    if (null != delta) {
      DoubleArrays.recycle(delta);
      delta = null;
    }
    if (null != deltaCompensation) {
      DoubleArrays.recycle(deltaCompensation);
      deltaCompensation = null;
    }
    super.finalize();
  }
  
  @Override
  public Delta<K> set(double[] data) {
    super.set(data);
    return this;
  }
  
  @Override
  public Delta<K> copy() {
    return new Delta(layer, target, DoubleArrays.copyOf(delta), DoubleArrays.copyOf(deltaCompensation));
  }
  
  /**
   * Add in place delta.
   *
   * @param buffer the buffer
   * @return the delta
   */
  public Delta<K> addInPlace(Delta<K> buffer) {
    return addInPlace(buffer.delta).addInPlace(buffer.deltaCompensation);
  }
}
