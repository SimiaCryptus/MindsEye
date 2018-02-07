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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * A calculation node, to be evaluated by a network once the inputs are available.
 */
@SuppressWarnings("serial")
final class InnerNode extends LazyResult {
  /**
   * The Created by.
   */
  @SuppressWarnings("unused")
  public final String[] createdBy = Util.currentStack();
  private final DAGNetwork dagNetwork;
  private final @NotNull DAGNode[] inputNodes;
  private NNLayer layer;
  
  /**
   * Instantiates a new Inner node.
   *
   * @param dagNetwork the dag network
   * @param layer      the key
   * @param inputNodes the input nodes
   */
  @SafeVarargs
  InnerNode(final DAGNetwork dagNetwork, final @NotNull NNLayer layer, final DAGNode... inputNodes) {
    this(dagNetwork, layer, UUID.randomUUID(), inputNodes);
  }
  
  /**
   * Instantiates a new Inner node.
   *
   * @param dagNetwork the dag network
   * @param layer      the layer
   * @param key        the key
   * @param inputNodes the input nodes
   */
  @SafeVarargs
  InnerNode(final DAGNetwork dagNetwork, final @NotNull NNLayer layer, final UUID key, final @NotNull DAGNode... inputNodes) {
    super(key);
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    setLayer(layer);
    this.inputNodes = inputNodes;
    assert Arrays.stream(inputNodes).parallel().allMatch(x -> x != null);
    for (@NotNull DAGNode node : this.inputNodes) {
      node.addRef();
    }
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(final NNLayer nextHead) {
    return dagNetwork.add(nextHead, InnerNode.this);
  }
  
  @Override
  protected NNResult eval(final GraphEvaluationContext ctx) {
    assertAlive();
    final @NotNull NNLayer innerLayer = getLayer();
    assert Arrays.stream(inputNodes).allMatch(x -> x != null);
    @NotNull Stream<DAGNode> stream = Arrays.stream(inputNodes);
    //stream = stream.parallel();
    final NNResult[] in = stream.map(x -> x == null ? null : x.get(ctx)).toArray(i -> new NNResult[i]);
    assert Arrays.stream(in).allMatch(x -> x != null);
    NNResult result = innerLayer.eval(in);
    for (@NotNull NNResult inputNNResult : in) {
      inputNNResult.getData().freeRef();
      inputNNResult.freeRef();
    }
    return result;
  }
  
  @Override
  public @NotNull DAGNode[] getInputs() {
    return inputNodes;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public @NotNull <T extends NNLayer> T getLayer() {
    return (T) layer;
  }
  
  @Override
  public synchronized void setLayer(final @NotNull NNLayer newLayer) {
    assertAlive();
    dagNetwork.assertAlive();
    synchronized (dagNetwork.layersById) {
      if (!dagNetwork.layersById.containsKey(newLayer.getId())) {
        NNLayer put = dagNetwork.layersById.put(newLayer.getId(), newLayer);
        newLayer.addRef();
        if (null != put) put.freeRef();
      }
    }
    newLayer.addRef();
    if (null != this.layer) this.layer.freeRef();
    this.layer = newLayer;
    dagNetwork.assertConsistent();
  }
  
  @Override
  public DAGNetwork getNetwork() {
    return dagNetwork;
  }
  
  @Override
  protected void _free() {
    super._free();
    for (@NotNull DAGNode node : this.inputNodes) {
      node.freeRef();
    }
    this.layer.freeRef();
  }
}
