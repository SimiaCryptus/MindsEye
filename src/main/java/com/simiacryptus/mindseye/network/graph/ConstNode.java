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
import com.simiacryptus.mindseye.layers.util.ConstNNLayer;
import com.simiacryptus.util.ml.Tensor;

import java.util.UUID;

public class ConstNode implements DAGNode {
  
  private final UUID id = UUID.randomUUID();
  private final ConstNNLayer layer;
  
  public ConstNode(Tensor tensor) {
    this.layer = new ConstNNLayer(id, tensor);
  }
  
  @Override
  public UUID getId() {
    return id;
  }
  
  @Override
  public NNLayer getLayer() {
    return layer;
  }
  
  @Override
  public NNResult get(EvaluationContext buildExeCtx) {
    return layer.eval(new NNResult[]{});
  }
}
