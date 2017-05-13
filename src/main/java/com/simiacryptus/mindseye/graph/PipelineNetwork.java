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

package com.simiacryptus.mindseye.graph;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.graph.dag.DAGNetwork;
import com.simiacryptus.mindseye.graph.dag.DAGNode;
import com.simiacryptus.mindseye.net.NNLayer;

import java.util.Arrays;
import java.util.HashMap;

public class PipelineNetwork extends DAGNetwork {
  
  private final HashMap<NNLayer, NNLayer> forwardLinkIndex = new HashMap<>();
  private final HashMap<NNLayer, NNLayer> backwardLinkIndex = new HashMap<>();
  private DAGNode head = getInput().get(0);
  
  public PipelineNetwork() {
    this(1);
  }
  
  public PipelineNetwork(int inputs) {
    super(inputs);
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    DAGNode node = super.add(nextHead, head);
    assert Arrays.stream(head).allMatch(x -> x != null);
    if (head.length > 0) {
      // XXX: Prev/next linking only tracks first input node
      final NNLayer prevHead = getLayer(head[0]);
      this.backwardLinkIndex.put(nextHead, prevHead);
      this.forwardLinkIndex.put(prevHead, nextHead);
    }
    assert null != getInput();
    setHead(node);
    return node;
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
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", getHead().toJson());
    return json;
  }
  
}
