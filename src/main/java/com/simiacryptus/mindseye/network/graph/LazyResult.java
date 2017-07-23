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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.NNResult;

import java.util.UUID;

abstract class LazyResult implements DAGNode {
  
  public final UUID key;
  
  public LazyResult() {
    this(UUID.randomUUID());
  }
  
  protected LazyResult(final UUID key) {
    super();
    this.key = key;
  }
  
  protected abstract NNResult eval(EvaluationContext t);
  
  @Override
  public NNResult get(final EvaluationContext t) {
    return t.cache.computeIfAbsent(this.key, k -> {
      try {
        return eval(t);
      } catch (Throwable e) {
        throw new RuntimeException("Error with layer " + getLayer(), e);
      }
    });
  }
  
}
