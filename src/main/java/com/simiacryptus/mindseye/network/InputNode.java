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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;

import javax.annotation.Nonnull;
import java.util.UUID;

/**
 * A node providing access to given inputs for LayerBase evaluation.
 */
@SuppressWarnings("serial")
final class InputNode extends LazyResult {
  private final DAGNetwork dagNetwork;

  /**
   * Instantiates a new Input node.
   *
   * @param dagNetwork the dag network
   */
  InputNode(final DAGNetwork dagNetwork) {
    this(dagNetwork, null);
  }

  /**
   * Instantiates a new Input node.
   *
   * @param dagNetwork the dag network
   * @param key        the key
   */
  public InputNode(final DAGNetwork dagNetwork, final UUID key) {
    super(key);
    this.dagNetwork = dagNetwork;
  }

  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @return the dag node
   */
  public DAGNode add(@Nonnull final Layer nextHead) {
    return dagNetwork.add(nextHead, InputNode.this);
  }

  @Override
  protected Result eval(@Nonnull final GraphEvaluationContext context) {
    assertAlive();
    this.dagNetwork.assertAlive();
    synchronized (context) {
      CountingResult countingNNResult = context.calculated.get(id).get();
      countingNNResult.addRef();
      return countingNNResult;
    }
  }

  @Override
  public <T extends Layer> T getLayer() {
    return null;
  }

  @Override
  public void setLayer(final Layer layer) {
    throw new IllegalStateException();
  }

  @Override
  public DAGNetwork getNetwork() {
    return this.dagNetwork;
  }

  @Override
  protected void _free() {
    super._free();
  }
}
