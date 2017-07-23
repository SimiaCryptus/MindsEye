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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.layers.util.ConstNNLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.util.ml.Tensor;

import java.util.*;
import java.util.stream.Collectors;

public class PipelineNetwork extends DAGNetwork {
  
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    List<UUID> layerIds = getLayers().stream().map(x -> x.getId()).collect(Collectors.toList());
    JsonArray layerArray = new JsonArray();
    getLayers().forEach(l->layerArray.add(new JsonPrimitive(l.getId().toString())));
    json.add("layerList", layerArray);
    return json;
  }
  
  public static PipelineNetwork fromJson(JsonObject json) {
    return new PipelineNetwork(json);
  }
  protected PipelineNetwork(JsonObject json) {
    super(json);
    UUID head = UUID.fromString(json.get("head").getAsString());
    this.head = nodesById.get(head);
    getLayers().clear();
    json.get("layerList").getAsJsonArray().forEach(element->{
      getLayers().add(getChild(UUID.fromString(element.getAsString())));
    });
    if(null == this.head) throw new IllegalArgumentException();
  }
  
  private DAGNode head;
  private List<NNLayer> layers;
  
  public PipelineNetwork() {
    this(1);
    head = getInput().get(0);
  }
  
  public PipelineNetwork(int inputs) {
    super(inputs);
    head = 0==inputs?null:getInput().get(0);
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    if(null == nextHead) throw new IllegalArgumentException();
    DAGNode node = super.add(nextHead, head);
    assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    if(getLayers().stream().filter(l->l.getId().equals(nextHead.getId())).findAny().isPresent()) throw new IllegalArgumentException(nextHead.getName() + " already added");
    getLayers().add(nextHead);
    return node;
  }
  
  public DAGNode constValue(Tensor tensor) {
    DAGNode constNode = super.add(new ConstNNLayer(tensor));;
    getLayers().add(constNode.getLayer());
    return constNode;
  }
  
  public DAGNode add(NNLayer nextHead) {
    DAGNode head = getHead();
    if (null == head) return add(nextHead, getInput(0));
    return add(nextHead, head);
  }
  
  public DAGNode getHead() {
    return this.head;
  }
  
  private void setHead(final DAGNode imageRMS) {
    this.head = imageRMS;
  }
  
  public List<NNLayer> getLayers() {
    if(null == layers) {
      synchronized(this) {
        if(null == layers) {
          layers = new ArrayList<>();
        }
      }
    }
    return layers;
  }
  
  public PipelineNetwork setLayers(List<NNLayer> layers) {
    this.layers = layers;
    return this;
  }
}
