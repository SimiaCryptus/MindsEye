/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.ValueLayer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/**
 * A simple network architecture based on the assumption of a linear sequence of components. Each component added
 * becomes the new head node, and a default add method appends a new node on the existing head.
 */
@SuppressWarnings("serial")
public class PipelineNetwork extends DAGNetwork {
  @Nullable
  private DAGNode head;
  
  /**
   * Instantiates a new Pipeline network.
   */
  public PipelineNetwork() {
    this(1);
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param inputs the inputs
   * @param layers the layers
   */
  public PipelineNetwork(final int inputs, @Nonnull final Layer... layers) {
    super(inputs);
    for (final Layer layer : layers) {
      add(layer).freeRef();
    }
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected PipelineNetwork(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    @Nonnull final UUID headId = UUID.fromString(json.get("head").getAsString());
    if (!inputHandles.contains(headId)) {
      assert null != headId;
      setHead(getNodeById(headId));
    }
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param layers the layers
   */
  public PipelineNetwork(final Layer... layers) {
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
  public static PipelineNetwork fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PipelineNetwork(json, rs);
  }
  
  /**
   * Build pipeline network.
   *
   * @param inputs the inputs
   * @param layers the layers
   * @return the pipeline network
   */
  public static PipelineNetwork build(final int inputs, final Layer... layers) {
    PipelineNetwork pipelineNetwork = new PipelineNetwork(inputs);
    for (final Layer layer : layers) {
      pipelineNetwork.add(layer).freeRef();
    }
    return pipelineNetwork;
  }
  
  public static PipelineNetwork wrap(final int inputs, final Layer... layers) {
    PipelineNetwork pipelineNetwork = new PipelineNetwork(inputs);
    for (final Layer layer : layers) {
      pipelineNetwork.wrap(layer).freeRef();
    }
    return pipelineNetwork;
  }
  
  @Nonnull
  @Override
  public PipelineNetwork copy(final SerialPrecision precision) {
    return (PipelineNetwork) super.copy(precision);
  }
  
  @Nonnull
  @Override
  public PipelineNetwork copy() {
    return (PipelineNetwork) super.copy();
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  @Nullable
  public InnerNode add(@Nullable final Layer nextHead) {
    assert nextHead.assertAlive();
    if (null == nextHead) return null;
    return add(nextHead, getHead());
  }
  
  /**
   * Wrap dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  @Nullable
  public InnerNode wrap(@Nullable final Layer nextHead) {
    @Nullable InnerNode add = add(nextHead);
    nextHead.freeRef();
    return add;
  }
  
  @Nullable
  public InnerNode wrap(@Nullable final Layer nextHead, @Nonnull final DAGNode... head) {
    @Nullable InnerNode add = add(nextHead, head);
    nextHead.freeRef();
    return add;
  }
  
  /**
   * Wrap dag node.
   *
   * @param label    the label
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  @Nullable
  public DAGNode wrap(final CharSequence label, @Nullable final Layer nextHead, @Nonnull final DAGNode... head) {
    DAGNode add = add(label, nextHead, head);
    nextHead.freeRef();
    return add;
  }
  
  @Nullable
  @Override
  public InnerNode add(@Nullable final Layer nextHead, @Nonnull final DAGNode... head) {
    if (null == nextHead && head.length > 0) throw new IllegalArgumentException();
    if (null == nextHead) return null;
    assert Arrays.stream(head).allMatch(x -> x == null || internalNodes.containsKey(x.getId()) || inputNodes.containsKey(x.getId()));
    @Nullable final InnerNode node = super.add(nextHead, head);
    assert null != inputHandles;
    setHead(node);
    return node;
  }
  
  @SafeVarargs
  @Override
  public final InnerNode add(final CharSequence label, @Nullable final Layer layer, final DAGNode... head) {
    if (null == layer) throw new IllegalArgumentException();
    final InnerNode node = super.add(label, layer, head);
    //assert Arrays.stream(head).allMatch(x -> x != null);
    assert null != inputHandles;
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
  public InnerNode addAll(InnerNode node, @Nonnull final Layer... layers) {
    for (final Layer l : layers) {
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
  public InnerNode addAll(final Layer... layers) {
    return addAll((InnerNode) getHead(), layers);
  }
  
  /**
   * Const value dag node.
   *
   * @param tensor the tensor
   * @return the dag node
   */
  @Nullable
  public DAGNode constValue(final Tensor tensor) {
    return super.wrap(new ValueLayer(tensor));
  }
  
  @Nullable
  public DAGNode constValueWrap(final Tensor tensor) {
    DAGNode node = constValue(tensor);
    tensor.freeRef();
    return node;
  }
  
  @Override
  protected void _free() {
    super._free();
    if (null != head) {
      head.freeRef();
      head = null;
    }
  }
  
  @Nullable
  @Override
  public DAGNode getHead() {
    if (null == head) {
      return getInput(0);
    }
    else {
      head.addRef();
      return head;
    }
  }
  
  /**
   * Sets head.
   *
   * @param obj the obj
   * @return the head
   */
  @Nonnull
  public PipelineNetwork setHead(final DAGNode obj) {
    if (obj != head) {
      if (null != head) head.freeRef();
      head = obj;
      if (null != head) head.addRef();
    }
    return this;
  }
  
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    assertConsistent();
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("head", getHeadId().toString());
    return json;
  }
  
}
