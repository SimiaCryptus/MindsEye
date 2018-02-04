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
  private final DAGNode[] inputNodes;
  private NNLayer layer;
  
  /**
   * Instantiates a new Inner node.
   *
   * @param dagNetwork the dag network
   * @param key        the key
   * @param inputNodes the input nodes
   */
  @SafeVarargs
  InnerNode(final DAGNetwork dagNetwork, final NNLayer key, final DAGNode... inputNodes) {
    this(dagNetwork, key, UUID.randomUUID(), inputNodes);
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
  InnerNode(final DAGNetwork dagNetwork, final NNLayer layer, final UUID key, final DAGNode... inputNodes) {
    super(key);
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    setLayer(layer);
    this.inputNodes = inputNodes;
    assert Arrays.stream(inputNodes).parallel().allMatch(x -> x != null);
    for (DAGNode node : this.inputNodes) {
      node.addRef();
    }
    this.layer.addRef();
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
    final NNLayer innerLayer = getLayer();
    assert Arrays.stream(inputNodes).allMatch(x -> x != null);
    Stream<DAGNode> stream = Arrays.stream(inputNodes);
    //stream = stream.parallel();
    final NNResult[] in = stream.map(x -> x == null ? null : x.get(ctx)).toArray(i -> new NNResult[i]);
    assert Arrays.stream(in).allMatch(x -> x != null);
    NNResult result = innerLayer.eval(in);
    for (NNResult inputNNResult : in) {
      inputNNResult.getData().freeRef();
      inputNNResult.freeRef();
    }
    return result;
  }
  
  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  }
  
  @SuppressWarnings("unchecked")
  @Override
  public <T extends NNLayer> T getLayer() {
    return (T) layer;
  }
  
  @Override
  public void setLayer(final NNLayer layer) {
    dagNetwork.layersById.put(layer.getId(), layer);
    this.layer = layer;
    dagNetwork.assertConsistent();
  }
  
  @Override
  public DAGNetwork getNetwork() {
    return dagNetwork;
  }
  
  @Override
  protected void _free() {
    for (DAGNode node : this.inputNodes) {
      node.freeRef();
    }
    this.layer.freeRef();
  }
}
