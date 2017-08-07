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

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.TensorList;

import java.util.ArrayList;
import java.util.List;

public class CountingNNResult extends NNResult {
  private int count = 0;
  
  protected CountingNNResult(NNResult inner) {
    super(inner.data);
    this.inner = inner;
  }

  public int getCount() {
    return count;
  }

  public CountingNNResult increment() {
    add(1);
    return this;
  }

  public synchronized int add(int count) {
    this.count += count;
    //System.err.println("Count -> " + this.count);
    return this.count;
  }
  final NNResult inner;
  final List<TensorList> passbackBuffer = new ArrayList<>();
  
  
  @Override
  public void accumulate(DeltaSet buffer, TensorList data) {
    passbackBuffer.add(data);
    if(passbackBuffer.size() >= getCount()) {
      //System.err.println(String.format("Pass Count -> %s, Buffer -> %s", this.count, passbackBuffer.size()));
      TensorList reduced = passbackBuffer.stream().reduce(TensorList::add).get();
      inner.accumulate(buffer, reduced);
      passbackBuffer.clear();
    } else {
      //System.err.println(String.format("Accum Count -> %s, Buffer -> %s", this.count, passbackBuffer.size()));
    }
    
  }
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
  
  
  
}
