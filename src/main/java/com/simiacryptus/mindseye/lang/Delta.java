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

public class Delta extends DoubleBuffer {
  protected double[] deltaCompensation;
  
  /**
   * Instantiates a new Delta.
   *
   * @param target the target
   * @param delta  the delta
   * @param layer  the layer
   */
  public Delta(final double[] target, final double[] delta, final NNLayer layer) {
    super(layer, target, delta);
    if (null == target) throw new IllegalArgumentException();
    //if(null == array) throw new IllegalArgumentException();
    this.deltaCompensation = DoubleArrays.obtain(delta.length);
  }
  
  /**
   * Instantiates a new Delta.
   *
   * @param target the target
   * @param layer  the layer
   */
  public Delta(final double[] target, final NNLayer layer) {
    this(target,DoubleArrays.obtain(target.length),layer);
  }
  
  /**
   * Accumulate.
   *
   * @param data  the data
   * @param delta the delta
   */
  public static void accumulate(double[] data, double[] delta, double[] dataCompensation) {
    for (int i = 0; i < data.length; i++) {
      double sum = data[i];
      double input = delta[i];
      double c = null==dataCompensation?0:dataCompensation[i];
      if(Math.abs(sum) >= Math.abs(input)) {
        double y = sum - c;
        double t = input + y;
        c = (t-input) - y;
        data[i] = t;
        if(null != dataCompensation) dataCompensation[i] = c;
      } else {
        double y = input - c;
        double t = sum + y;
        c = (t-sum) - y;
        data[i] = t;
        if(null != dataCompensation) dataCompensation[i] = c;
      }
    }
  }
  
  /**
   * Accumulate delta.
   *
   * @param data the data
   * @return the delta
   */
  public DoubleBuffer accumulate(final double[] data) {
    //assert Arrays.stream(data).allMatch(Double::isFinite);
    accumulate(delta, data, deltaCompensation);
    //assert Arrays.stream(getDelta()).allMatch(Double::isFinite);
    return this;
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
   * Scale delta.
   *
   * @param f the f
   * @return the delta
   */
  public Delta scale(final double f) {
    return map(x -> x * f);
  }
  
  public Delta map(final DoubleUnaryOperator mapper) {
    return new Delta(this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
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
  
  @Override
  protected void finalize() throws Throwable {
    if(null != delta) {
      DoubleArrays.recycle(delta);
      delta = null;
    }
    if(null != deltaCompensation) {
      DoubleArrays.recycle(deltaCompensation);
      deltaCompensation = null;
    }
    super.finalize();
  }
  
  @Override
  public Delta set(double[] data) {
    super.set(data);
    return this;
  }
  
  @Override
  public Delta copy() {
    return new Delta(target, DoubleArrays.copyOf(delta), layer);
  }
}
