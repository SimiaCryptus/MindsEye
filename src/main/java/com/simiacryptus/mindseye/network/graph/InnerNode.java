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

package com.simiacryptus.mindseye.network.graph;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.UUID;

/**
 * The type Inner node.
 */
public final class InnerNode extends LazyResult {
  /**
   * The Layer.
   */
  public final NNLayer layer;
  /**
   * The Created by.
   */
  @SuppressWarnings("unused")
  public final String[] createdBy = Util.currentStack();
  /**
   * The Id.
   */
  public final UUID id;
  private final DAGNetwork dagNetwork;
  private final DAGNode[] inputNodes;
  
  /**
   * Instantiates a new Inner node.
   *
   * @param dagNetwork the dag network
   * @param id         the id
   * @param inputNodes the input nodes
   */
  @SafeVarargs
  InnerNode(DAGNetwork dagNetwork, final NNLayer id, final DAGNode... inputNodes) {
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    this.id = id.getId();
    this.layer = id;
    assert Arrays.stream(inputNodes).allMatch(x -> x != null);
    this.inputNodes = inputNodes;
  }
  
  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  }
  
  @Override
  protected NNResult eval(final EvaluationContext ctx, NNLayer.NNExecutionContext nncontext) {
    if (1 == this.inputNodes.length) {
      final NNResult in = this.inputNodes[0].get(nncontext, ctx);
      final NNResult output = dagNetwork.layersById.get(this.id).eval(nncontext, in);
      return output;
    } else {
      final NNResult[] in = Arrays.stream(this.inputNodes).map(x -> x.get(nncontext, ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = dagNetwork.layersById.get(this.id).eval(nncontext, in);
      return output;
    }
  }
  
  @Override
  public UUID getId() {
    return this.id;
  }
  
  @Override
  public NNLayer getLayer() {
    return layer;
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
