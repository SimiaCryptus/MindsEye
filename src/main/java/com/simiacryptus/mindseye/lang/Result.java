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

import java.util.Arrays;
import java.util.function.BiConsumer;

/**
 * Encapsulates the results of evaluating neural network. It includes both the result data and a function which can be
 * evaluated to determine the learning gradient. Does not hold a reference on the result data object, allowing that data
 * to be freed when possible while preserving the gradient callback.
 */
public class Result extends ReferenceCountingBase {
  /**
   * The Data.
   */
  protected final TensorList data;
  /**
   * The Accumulator.
   */
  protected final BiConsumer<DeltaSet<Layer>, TensorList> accumulator;
  
  /**
   * Instantiates a new Nn result.
   *
   * @param data        the data
   * @param accumulator the accumulator
   */
  public Result(final TensorList data, BiConsumer<DeltaSet<Layer>, TensorList> accumulator) {
    super();
    this.data = data;
    this.accumulator = accumulator;
  }
  
  /**
   * Get single delta double [ ].
   *
   * @return the double [ ]
   */
  public double[] getSingleDelta() {
    DeltaSet<Layer> deltaBuffer = new DeltaSet<>();
    accumulate(deltaBuffer);
    if (deltaBuffer.getMap().size() != 1) throw new AssertionError(deltaBuffer.getMap().size());
    double[] delta = copy(deltaBuffer.getMap().values().iterator().next().getDelta());
    deltaBuffer.freeRef();
    return delta;
  }
  
  public double[] copy(double[] delta) {
    delta = Arrays.copyOf(delta, delta.length);
    return delta;
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   */
  public final void accumulate(final DeltaSet<Layer> buffer) {
    accumulate(buffer, 1.0);
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param value  the value
   */
  public final void accumulate(final DeltaSet<Layer> buffer, final double value) {
    accumulate(buffer, TensorArray.wrap(getData().stream().map(t -> t.mapAndFree(v -> value)).toArray(i -> new Tensor[i])));
  }
  
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param delta  the evalInputDelta
   */
  public void accumulate(DeltaSet<Layer> buffer, TensorList delta) {
    try {
      getAccumulator().accept(buffer, delta);
    } finally {
      delta.freeRef();
    }
  }
  
  /**
   * Gets data.
   *
   * @return the data
   */
  public final TensorList getData() {
    return data;
  }
  
  /**
   * Is alive boolean.
   *
   * @return the boolean
   */
  public boolean isAlive() {
    return null != getAccumulator();
  }
  
  /**
   * Gets accumulator.
   *
   * @return the accumulator
   */
  public BiConsumer<DeltaSet<Layer>, TensorList> getAccumulator() {
    assertAlive();
    return accumulator;
  }
  
  /**
   * Gets data and free.
   *
   * @return the data and free
   */
  public TensorList getDataAndFree() {
    assertAlive();
    TensorList data = getData();
    data.assertAlive();
    freeRef();
    return data;
  }
}
