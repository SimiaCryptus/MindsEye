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

import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.meta.Sparse01MetaLayer;
import com.simiacryptus.mindseye.layers.reducers.SumReducerLayer;

public class SparseAutoencoderTrainer extends SupervisedNetwork {
  
  public final DAGNode encoder;
  public final DAGNode decoder;
  public final DAGNode loss;
  public final DAGNode sparsity;
  public final DAGNode sumSparsityLayer;
  public final DAGNode sumFitnessLayer;
  public final DAGNode sparsityThrottleLayer;
  
  public SparseAutoencoderTrainer(final NNLayer encoder, final NNLayer decoder) {
    super(1);
    this.encoder = add(encoder, getInput(0));
    this.decoder = add(decoder, this.encoder);
    this.loss = add(new MeanSqLossLayer(), this.decoder, getInput(0));
    this.sparsity = add(new Sparse01MetaLayer(), this.encoder);
    this.sumSparsityLayer = add(new SumReducerLayer(), this.sparsity);
    this.sparsityThrottleLayer = add(new LinearActivationLayer().setScale(0.5), this.sumSparsityLayer);
    this.sumFitnessLayer = add(new SumReducerLayer(), this.sparsityThrottleLayer, this.loss);
  }
  
  @Override
  public DAGNode getHead() {
    return this.sumFitnessLayer;
  }
}
