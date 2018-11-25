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
import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * Alternate version being staged to effect an in-memory change to a double[] array. In comparison apply the Delta class
 * via geometric analogy, this would be a point whereas Delta is a vector.
 *
 * @param <K> the type parameter
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class State<K> extends DoubleBuffer<K> {


  /**
   * Instantiates a new State.
   *
   * @param layer  the key
   * @param target the target
   */
  public State(@Nonnull final K layer, final double[] target) {
    super(layer, target);
  }

  /**
   * Instantiates a new Delta.
   *
   * @param layer  the key
   * @param target the target
   * @param delta  the evalInputDelta
   */
  public State(@Nonnull final K layer, final double[] target, final double[] delta) {
    super(layer, target, delta);
  }

  /**
   * Are equal boolean.
   *
   * @return the boolean
   */
  public boolean areEqual() {
    return DoubleBuffer.areEqual(getDelta(), target);
  }

  /**
   * Backup double buffer.
   *
   * @return the double buffer
   */
  @Nonnull
  public final synchronized State<K> backup() {
    System.arraycopy(target, 0, getDelta(), 0, target.length);
    return this;
  }

  @Nonnull
  @Override
  public State<K> copy() {
    assertAlive();
    return new State(key, target, RecycleBin.DOUBLES.copyOf(delta, length()));
  }

  /**
   * Backup copy state.
   *
   * @return the state
   */
  @Nonnull
  public State<K> backupCopy() {
    return new State(key, target, RecycleBin.DOUBLES.copyOf(target, length()));
  }

  @Nonnull
  @Override
  public State<K> map(@Nonnull final DoubleUnaryOperator mapper) {
    return new State(key, target, Arrays.stream(getDelta()).map(x -> mapper.applyAsDouble(x)).toArray());
  }

  /**
   * Overwrite.
   *
   * @return the double buffer
   */
  @Nonnull
  public final synchronized State<K> restore() {
    System.arraycopy(getDelta(), 0, target, 0, target.length);
    return this;
  }

}
