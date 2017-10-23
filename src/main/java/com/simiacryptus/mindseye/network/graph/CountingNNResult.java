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

package com.simiacryptus.mindseye.network.graph;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.f32.CuDNNFloatTensorList;

/**
 * The type Counting nn result.
 */
public class CountingNNResult extends NNResult {
  private int count = 0;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner the inner
   */
  protected CountingNNResult(NNResult inner) {
    super(inner.getData());
    this.inner = inner;
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
  
  /**
   * Add int.
   *
   * @param count the count
   * @return the int
   */
  public synchronized int add(int count) {
    this.count += count;
    //System.err.println("Count -> " + this.count);
    return this.count;
  }
  
  /**
   * The Inner.
   */
  final NNResult inner;
  /**
   * The Passback buffer.
   */
  TensorList passbackBuffer = null;
  /**
   * The Queued.
   */
  int queued = 0;
  
  
  @Override
  public void accumulate(DeltaSet buffer, TensorList data) {
    if (data instanceof CuDNNFloatTensorList) assert null != ((CuDNNFloatTensorList) data).ptr.getPtr();
    if (null == passbackBuffer) {
      if (1 == getCount()) {
        passbackBuffer = data;
      }
      else {
        passbackBuffer = data.copy();
      }
    }
    else {
      if (passbackBuffer.length() == 0) {
        passbackBuffer = data;
      }
      else {
        passbackBuffer.accum(data);
      }
    }
    if (++queued == getCount()) {
      //System.err.println(String.format("Pass Count -> %s, Buffer -> %s", this.count, passbackBuffer.size()));
      inner.accumulate(buffer, passbackBuffer);
      queued = 0;
    }
    else {
      //System.err.println(String.format("Accum Count -> %s, Buffer -> %s", this.count, passbackBuffer.size()));
    }
    
  }
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
  
  @Override
  protected void finalize() throws Throwable {
    super.finalize();
  }
}
