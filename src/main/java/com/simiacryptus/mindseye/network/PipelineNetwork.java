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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.network.graph.ConstNode;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.UUID;

public class PipelineNetwork extends DAGNetwork {
  
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    return json;
  }
  
  public static PipelineNetwork fromJson(JsonObject json) {
    return new PipelineNetwork(json);
  }
  protected PipelineNetwork(JsonObject json) {
    super(json);
    UUID head = UUID.fromString(json.get("head").getAsString());
    this.head = nodesById.get(head);
    if(null == this.head) throw new IllegalArgumentException();
  }
  
  private DAGNode head;
  
  public PipelineNetwork() {
    this(1);
    head = getInput().get(0);
  }
  
  public PipelineNetwork(int inputs) {
    super(inputs);
    head = getInput().get(0);
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    DAGNode node = super.add(nextHead, head);
    assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  public DAGNode constValue(Tensor tensor) {
    return new ConstNode(tensor);
  }
  
  public DAGNode add(NNLayer nextHead) {
    DAGNode head = getHead();
    if (null == head) return add(nextHead, getInput(0));
    return add(nextHead, head);
  }
  
  public DAGNode getHead() {
    return this.head;
  }
  
  public void setHead(final DAGNode imageRMS) {
    this.head = imageRMS;
  }
  
}
