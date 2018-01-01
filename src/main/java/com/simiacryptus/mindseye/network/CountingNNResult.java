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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;

/**
 * A result type for evaluating the backpropigation phase of an Acyclic Directed Graph. Since the result of a given
 * compoent in a network can be used multiple times, we can improve efficiency of backprop by accumulating all the
 * backpropigating delta signals into a single signal before evaluating further backwards.
 */
class CountingNNResult extends NNResult {
  
  /**
   * The Queued.
   */
  int accumulations = 0;

  /**
   * The Inner.
   */
  final NNResult inner;
  /**
   * The Passback buffer.
   */
  TensorList passbackBuffer = null;
  int finalizations = 0;

  /**
   * A flagrant abuse of Java's object finalization contract.
   * Repeated calls to this class's finalize method will increment a counter,
   * and when the counter cycles the call is chained.
   */
  @Override
  public void finalize() {
    if (1 >= getCount()) {
      inner.finalize();
    }
    else {
      if (++finalizations == getCount()) {
        inner.finalize();
        finalizations = 0;
      }
    }
  }
  private int count = 0;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner the inner
   */
  protected CountingNNResult(final NNResult inner) {
    super(inner.getData());
    this.inner = inner;
  }
  
  @Override
  public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
    if (1 >= getCount()) {
      inner.accumulate(buffer, data);
    }
    else {
      if (null == passbackBuffer) {
        passbackBuffer = data.copy();
      }
      else {
        passbackBuffer.accum(data);
      }
      if (++accumulations == getCount()) {
        inner.accumulate(buffer, passbackBuffer);
        if (1 < getCount()) {
          passbackBuffer.recycle();
        }
        accumulations = 0;
      }
    }
  }
  
  /**
   * Add int.
   *
   * @param count the count
   * @return the int
   */
  public synchronized int add(final int count) {
    this.count += count;
    //System.err.println("Count -> " + this.count);
    return this.count;
  }
  
  /**
   * Gets count.
   *
   * @return the count
   */
  public int getCount() {
    return count;
  }
  
  /**
   * Increment counting nn result.
   *
   * @return the counting nn result
   */
  public CountingNNResult increment() {
    add(1);
    return this;
  }
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
}
