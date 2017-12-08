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

import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.UUID;

/**
 * A calculation node, to be evaluated by a network once the inputs are available.
 */
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
  InnerNode(DAGNetwork dagNetwork, final NNLayer key, final DAGNode... inputNodes) {
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
  InnerNode(DAGNetwork dagNetwork, final NNLayer layer, UUID key, final DAGNode... inputNodes) {
    super(key);
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    this.setLayer(layer);
    this.inputNodes = inputNodes;
  }
  
  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  }
  
  @Override
  protected NNResult eval(final GraphEvaluationContext ctx, NNExecutionContext nncontext) {
    NNLayer innerLayer = getLayer();
    if (1 == this.inputNodes.length) {
      DAGNode inputNode = this.inputNodes[0];
      final NNResult in = null == inputNode ? null : inputNode.get(nncontext, ctx);
      final NNResult output = innerLayer.eval(nncontext, in);
      return output;
    }
    else {
      final NNResult[] in = Arrays.stream(this.inputNodes).map(x -> x == null ? null : x.get(nncontext, ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = innerLayer.eval(nncontext, in);
      return output;
    }
  }
  
  @Override
  public NNLayer getLayer() {
    return layer;
  }
  
  public void setLayer(NNLayer layer) {
    this.dagNetwork.layersById.put(layer.getId(), layer);
    this.layer = layer;
    this.dagNetwork.assertConsistent();
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(NNLayer nextHead) {
    return dagNetwork.add(nextHead, InnerNode.this);
  }
}
