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
 * Alternate version being staged to effect an in-memory change to a double[] array.
 * In comparison with the Delta class via geometric analogy, this would be a point whereas Delta is a vector.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class State extends DoubleBuffer {
  
  
  /**
   * Instantiates a new Delta.
   *
   * @param target the target
   * @param delta  the delta
   * @param layer  the layer
   */
  public State(final double[] target, final double[] delta, final NNLayer layer) {
    super(target,delta,layer);
  }
  
  public State(final double[] target, final NNLayer layer) {
    super(target,layer);
  }
  
  public final synchronized DoubleBuffer backup() {
    double[] delta = getDelta();
    System.arraycopy(target,0,delta,0,target.length);
    return this;
  }
  
  /**
   * Overwrite.
   */
  public final synchronized DoubleBuffer restore() {
    System.arraycopy(delta,0,target,0,target.length);
    return this;
  }
  
  public State accumulate(double[] delta) {
    Delta.accumulate(this.delta, delta, null);
    return this;
  }
  
  public State(NNLayer layer, double[] target, double[] delta) {
    super(layer, target, delta);
  }
  
  @Override
  public DoubleBuffer map(DoubleUnaryOperator mapper) {
    return new State(this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray(), this.layer);
  }
  
  @Override
  public DoubleBuffer copy() {
    return new State(target, DoubleArrays.copyOf(delta), layer);
  }
}