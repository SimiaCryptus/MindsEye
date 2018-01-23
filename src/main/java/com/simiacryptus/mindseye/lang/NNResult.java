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

public abstract class NNResult {
  /**
   * The Data.
   */
  protected final TensorList data;
  protected final BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator;
  
  public NNResult(BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator, final TensorList data) {
    super();
    this.data = data;
    this.accumulator = accumulator;
  }
  
  public NNResult(BiConsumer<DeltaSet<NNLayer>, TensorList> accumulator, final Tensor... data) {
    super();
    this.data = new TensorArray(data);
    this.accumulator = accumulator;
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
    final Tensor[] defaultVector = getData().stream().map(t -> t.map(v -> value)).toArray(i -> new Tensor[i]);
    accumulate(buffer, new TensorArray(defaultVector));
  }
  
  
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
  
  public BiConsumer<DeltaSet<NNLayer>, TensorList> getAccumulator() {
    return accumulator;
  }
}
