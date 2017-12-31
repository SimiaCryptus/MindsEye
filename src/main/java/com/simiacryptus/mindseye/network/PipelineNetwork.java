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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.ConstNNLayer;

import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/**
 * A simple network architecture based on the assumption of a linear sequence of components. Each component added
 * becomes the new head node, and a default add method appends a new node on the existing head.
 */
@SuppressWarnings("serial")
public class PipelineNetwork extends DAGNetwork {
  
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
   *
   * @param inputs the inputs
   * @param layers the layers
   */
  public PipelineNetwork(final int inputs, final NNLayer... layers) {
    super(inputs);
    head = 0 == inputs ? null : getInput().get(0);
    for (final NNLayer layer : layers) {
      add(layer);
    }
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected PipelineNetwork(final JsonObject json, Map<String, byte[]> rs) {
    super(json, rs);
    final UUID headId = UUID.fromString(json.get("head").getAsString());
    assert null != headId;
    head = nodesById.get(headId);
    if (null == head) {
      head = getInput().get(0);
    }
    final int inputIndex = inputHandles.indexOf(headId);
    if (null == head && 0 <= inputIndex) {
      head = getInput(inputIndex);
    }
    if (null == head) throw new IllegalArgumentException();
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param layers the layers
   */
  public PipelineNetwork(final NNLayer... layers) {
    this();
    addAll(layers);
  }
  
  /**
   * From json pipeline network.
   *
   * @param json the json
   * @param rs   the rs
   * @return the pipeline network
   */
  public static PipelineNetwork fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new PipelineNetwork(json, rs);
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(final NNLayer nextHead) {
    if (null == nextHead) return null;
    return add(nextHead, getHead());
  }
  
  @Override
  public DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    if (null == nextHead && head.length == 1) return head[0];
    if (null == nextHead) return null;
    assert Arrays.stream(head).allMatch(x -> x == null || nodesById.containsKey(x.getId()) || inputNodes.containsKey(x.getId()));
    final DAGNode node = super.add(nextHead, head);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  @SafeVarargs
  @Override
  public final DAGNode add(final String label, final NNLayer layer, final DAGNode... head) {
    if (null == layer) throw new IllegalArgumentException();
    final DAGNode node = super.add(label, layer, head);
    //assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != getInput();
    setHead(node);
    return node;
  }
  
  /**
   * Add all dag node.
   *
   * @param node   the node
   * @param layers the layers
   * @return the dag node
   */
  public DAGNode addAll(DAGNode node, final NNLayer... layers) {
    for (final NNLayer l : layers) {
      node = add(l, node);
    }
    return node;
  }
  
  /**
   * Add all dag node.
   *
   * @param layers the layers
   * @return the dag node
   */
  public DAGNode addAll(final NNLayer... layers) {
    return addAll(getHead(), layers);
  }
  
  /**
   * Const value dag node.
   *
   * @param tensor the tensor
   * @return the dag node
   */
  public DAGNode constValue(final Tensor tensor) {
    final DAGNode constNode = super.add(new ConstNNLayer(tensor));
    return constNode;
  }
  
  @Override
  public DAGNode getHead() {
    if (null == head) {
      head = getInput().get(0);
    }
    return head;
  }
  
  /**
   * Sets head.
   *
   * @param obj the obj
   * @return the head
   */
  public PipelineNetwork setHead(final DAGNode obj) {
    head = obj;
    return this;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    assertConsistent();
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("head", head.getId().toString());
    return json;
  }
  
}
