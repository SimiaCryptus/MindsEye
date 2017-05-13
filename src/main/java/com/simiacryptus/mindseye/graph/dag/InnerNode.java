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
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.UUID;

public final class InnerNode extends LazyResult {
  public final NNLayer layer;
  @SuppressWarnings("unused")
  public final String[] createdBy = Util.currentStack();
  public final UUID id;
  private final DAGNetwork dagNetwork;
  private final DAGNode[] inputNodes;
  
  @SafeVarargs
  InnerNode(DAGNetwork dagNetwork, final NNLayer id, final DAGNode... inputNodes) {
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    this.id = id.getId();
    this.layer = id;
    assert Arrays.stream(inputNodes).allMatch(x -> x != null);
    this.inputNodes = inputNodes;
  }
  
  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  }
  
  @Override
  protected NNResult eval(final EvaluationContext ctx) {
    if (1 == this.inputNodes.length) {
      final NNResult in = this.inputNodes[0].get(ctx);
      final NNResult output = dagNetwork.byId.get(this.id).eval(in);
      return output;
    } else {
      final NNResult[] in = Arrays.stream(this.inputNodes).map(x -> x.get(ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = dagNetwork.byId.get(this.id).eval(in);
      return output;
    }
  }
  
  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.add("id", dagNetwork.byId.get(this.id).getJson());
    if (this.inputNodes.length > 0) json.add("prev0", this.inputNodes[0].toJson());
    return json;
  }
  
  @Override
  public UUID getId() {
    return this.id;
  }
  
  @Override
  public NNLayer getLayer() {
    return layer;
  }
  
  public DAGNode add(NNLayer nextHead) {
    return dagNetwork.add(nextHead, InnerNode.this);
  }
}
