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
import com.simiacryptus.mindseye.layers.util.ConstNNLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.util.ml.Tensor;

import java.util.*;

/**
 * The type Pipeline network.
 */
public class PipelineNetwork extends DAGNetwork {
  
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    return json;
  }
  
  /**
   * From json pipeline network.
   *
   * @param json the json
   * @return the pipeline network
   */
  public static PipelineNetwork fromJson(JsonObject json) {
    return new PipelineNetwork(json);
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param json the json
   */
  protected PipelineNetwork(JsonObject json) {
    super(json);
    UUID head = UUID.fromString(json.get("head").getAsString());
    this.head = nodesById.get(head);
    int inputIndex = inputHandles.indexOf(head);
    if(null == this.head && 0<=inputIndex) {
      this.head = getInput(inputIndex);
    }
    if(null == this.head) throw new IllegalArgumentException();
  }
  
  private DAGNode head;
  
  /**
   * Instantiates a new Pipeline network.
   */
  public PipelineNetwork() {
    this(1);
    head = getInput().get(0);
  }
  
  /**
   * Instantiates a new Pipeline network.
   */
  public PipelineNetwork(NNLayer... layers) {
    this();
    addAll(layers);
  }
  
  public DAGNode addAll(NNLayer... layers) {
    return addAll(getHead(), layers);
  }
  
  public DAGNode addAll(DAGNode node, NNLayer... layers) {
    for(NNLayer l : layers) node = add(l, node);
    return node;
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param inputs the inputs
   */
  public PipelineNetwork(int inputs) {
    super(inputs);
    head = 0==inputs?null:getInput().get(0);
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(String label, final NNLayer nextHead, final DAGNode... head) {
    if(null == nextHead) throw new IllegalArgumentException();
    DAGNode node = super.add(label, nextHead, head);
    assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  @Override
  public DAGNode add(NNLayer nextHead, DAGNode... head) {
    if(null == nextHead && head.length==1) return head[0];
    if(null == nextHead) throw new IllegalArgumentException();
    DAGNode node = super.add(nextHead, head);
    assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  /**
   * Const value dag node.
   *
   * @param tensor the tensor
   * @return the dag node
   */
  public DAGNode constValue(Tensor tensor) {
    DAGNode constNode = super.add(new ConstNNLayer(tensor));;
    return constNode;
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(NNLayer nextHead) {
    DAGNode head = getHead();
    if (null == head) return add(nextHead, getInput(0));
    else return add(nextHead, head);
  }
  
  public DAGNode getHead() {
    return this.head;
  }
  
  public PipelineNetwork setHead(final DAGNode obj) {
    this.head = obj;
    return this;
  }

}
