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
 *
 * @param <K> the type parameter
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class State<K> extends DoubleBuffer<K> {
  
  
  /**
   * Instantiates a new Delta.
   *
   * @param layer  the layer
   * @param target the target
   * @param delta  the delta
   */
  public State(final K layer, final double[] target, final double[] delta) {
    super(layer, target, delta);
  }
  
  /**
   * Instantiates a new State.
   *
   * @param layer  the layer
   * @param target the target
   */
  public State(final K layer, final double[] target) {
    super(layer, target);
  }
  
  /**
   * Backup double buffer.
   *
   * @return the double buffer
   */
  public final synchronized State<K> backup() {
    System.arraycopy(target, 0, getDelta(), 0, target.length);
    return this;
  }
  
  /**
   * Overwrite.
   *
   * @return the double buffer
   */
  public final synchronized State<K> restore() {
    System.arraycopy(getDelta(), 0, target, 0, target.length);
    return this;
  }
  
  @Override
  public State<K> map(DoubleUnaryOperator mapper) {
    return new State(this.layer, this.target, Arrays.stream(this.getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }
  
  @Override
  public State<K> copy() {
    return new State(layer, target, DoubleArrays.copyOf(delta));
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
