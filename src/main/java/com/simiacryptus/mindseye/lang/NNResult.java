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

import java.util.function.BiConsumer;

/**
 * The type Nn result.
 */
public abstract class NNResult {
  /**
   * The Data.
   */
  protected final TensorList data;
  /**
   * The Accumulator.
   */
  protected final BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator;
  
  /**
   * Instantiates a new Nn result.
   *
   * @param data        the data
   * @param accumulator the accumulator
   */
  public NNResult(final TensorList data, BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator) {
    super();
    this.data = data;
    this.accumulator = accumulator;
  }
  
  /**
   * Instantiates a new Nn result.
   *
   * @param accumulator the accumulator
   * @param data        the data
   */
  public NNResult(BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator, final Tensor... data) {
    this(new TensorArray(data), accumulator);
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   */
  public final void accumulate(final DeltaSet<NNLayer> buffer) {
    accumulate(buffer, 1.0);
  }
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param value  the value
   */
  public final void accumulate(final DeltaSet<NNLayer> buffer, final double value) {
    TensorArray tensorArray = new TensorArray(getData().stream().map(t -> t.map(v -> value)).toArray(i -> new Tensor[i]));
    accumulate(buffer, tensorArray);
    tensorArray.freeRef();
  }
  
  
  /**
   * Accumulate.
   *
   * @param buffer the buffer
   * @param delta  the delta
   */
  public final void accumulate(DeltaSet<NNLayer> buffer, TensorList delta) {
    getAccumulator().accept(buffer, delta);
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
  public abstract boolean isAlive();
  
  /**
   * Free.
   */
  public void free() {}
  
  /**
   * Gets accumulator.
   *
   * @return the accumulator
   */
  public BiConsumer<DeltaSet<NNLayer>, TensorList> getAccumulator() {
    return accumulator;
  }
}
