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

package com.simiacryptus.mindseye.network.util;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.Sparse01MetaLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.SupervisedNetwork;

/**
 * The type Sparse autoencoder trainer.
 */
@SuppressWarnings("serial")
public class SparseAutoencoderTrainer extends SupervisedNetwork {
  
  /**
   * The Decoder.
   */
  public final DAGNode decoder;
  /**
   * The Encoder.
   */
  public final DAGNode encoder;
  /**
   * The Loss.
   */
  public final DAGNode loss;
  /**
   * The Sparsity.
   */
  public final DAGNode sparsity;
  /**
   * The Sparsity throttle layer.
   */
  public final DAGNode sparsityThrottleLayer;
  /**
   * The Sum fitness layer.
   */
  public final DAGNode sumFitnessLayer;
  /**
   * The Sum sparsity layer.
   */
  public final DAGNode sumSparsityLayer;
  
  /**
   * Instantiates a new Sparse autoencoder trainer.
   *
   * @param encoder the encoder
   * @param decoder the decoder
   */
  public SparseAutoencoderTrainer(final NNLayer encoder, final NNLayer decoder) {
    super(1);
    this.encoder = add(encoder, getInput(0));
    this.decoder = add(decoder, this.encoder);
    loss = add(new MeanSqLossLayer(), this.decoder, getInput(0));
    sparsity = add(new Sparse01MetaLayer(), this.encoder);
    sumSparsityLayer = add(new SumReducerLayer(), sparsity);
    sparsityThrottleLayer = add(new LinearActivationLayer().setScale(0.5), sumSparsityLayer);
    sumFitnessLayer = add(new SumReducerLayer(), sparsityThrottleLayer, loss);
  }
  
  @Override
  public DAGNode getHead() {
    return sumFitnessLayer;
  }
}
