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

package com.simiacryptus.mindseye.graph.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

final class InputNode extends LazyResult {
  public final UUID handle;
  private final DAGNetwork dagNetwork;
  
  InputNode(DAGNetwork dagNetwork) {
    this(dagNetwork, null);
  }
  
  public InputNode(DAGNetwork dagNetwork, final UUID handle) {
    super(handle);
    this.dagNetwork = dagNetwork;
    this.handle = handle;
  }
  
  @Override
  protected NNResult eval(final EvaluationContext t) {
    return t.cache.get(this.handle);
  }
  
  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("target", dagNetwork.inputHandles.toString());
    return json;
  }
  
  @Override
  public UUID getId() {
    return handle;
  }
  
  @Override
  public NNLayer getLayer() {
    return null;
  }
  
  public DAGNode add(NNLayer nextHead) {
    return dagNetwork.add(nextHead, InputNode.this);
  }
}
