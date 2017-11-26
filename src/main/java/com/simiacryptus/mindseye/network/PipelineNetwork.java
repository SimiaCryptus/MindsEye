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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.ConstNNLayer;

import java.util.Arrays;
import java.util.UUID;

/**
 * A simple network architecture based on the assumption of a linear sequence of components.
 * Each component added becomes the new head node, and a default add method
 * appends a new node on the existing head.
 */
public class PipelineNetwork extends DAGNetwork {
  
  private DAGNode head;
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param json the json
   */
  protected PipelineNetwork(JsonObject json) {
    super(json);
    UUID headId = UUID.fromString(json.get("head").getAsString());
    assert (null != headId);
    this.head = nodesById.get(headId);
    if (null == this.head) head = getInput().get(0);
    int inputIndex = inputHandles.indexOf(headId);
    if (null == this.head && 0 <= inputIndex) {
      this.head = getInput(inputIndex);
    }
    if (null == this.head) throw new IllegalArgumentException();
  }
  
  /**
   * Instantiates a new Pipeline network.
   */
  public PipelineNetwork() {
    this(1);
    head = getInput().get(0);
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param layers the layers
   */
  public PipelineNetwork(NNLayer... layers) {
    this();
    addAll(layers);
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param inputs the inputs
   * @param layers the layers
   */
  public PipelineNetwork(int inputs, NNLayer... layers) {
    super(inputs);
    head = 0 == inputs ? null : getInput().get(0);
    for (NNLayer layer : layers) add(layer);
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
  
  public JsonObject getJson() {
    assertConsistent();
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    assert null != NNLayer.fromJson(json) : "Smoke test deserialization";
    return json;
  }
  
  /**
   * Add all dag node.
   *
   * @param layers the layers
   * @return the dag node
   */
  public DAGNode addAll(NNLayer... layers) {
    return addAll(getHead(), layers);
  }
  
  /**
   * Add all dag node.
   *
   * @param node   the node
   * @param layers the layers
   * @return the dag node
   */
  public DAGNode addAll(DAGNode node, NNLayer... layers) {
    for (NNLayer l : layers) node = add(l, node);
    return node;
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(String label, final NNLayer layer, final DAGNode... head) {
    if (null == layer) throw new IllegalArgumentException();
    DAGNode node = super.add(label, layer, head);
    //assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  @Override
  public DAGNode add(NNLayer nextHead, DAGNode... head) {
    if (null == nextHead && head.length == 1) return head[0];
    if (null == nextHead) throw new IllegalArgumentException();
    assert Arrays.stream(head).allMatch(x -> x == null || nodesById.containsKey(x.getId()) || inputNodes.containsKey(x.getId()));
    DAGNode node = super.add(nextHead, head);
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
    DAGNode constNode = super.add(new ConstNNLayer(tensor));
    return constNode;
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(NNLayer nextHead) {
    return add(nextHead, getHead());
  }
  
  public DAGNode getHead() {
    if (null == this.head) head = getInput().get(0);
    return this.head;
  }
  
  /**
   * Sets head.
   *
   * @param obj the obj
   * @return the head
   */
  public PipelineNetwork setHead(final DAGNode obj) {
    this.head = obj;
    return this;
  }
  
}
