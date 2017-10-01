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

package com.simiacryptus.mindseye.layers.activation;

import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.NNLayer;

/**
 * The type Activation layer test base.
 */
public abstract class ActivationLayerTestBase extends LayerTestBase {
  
  /**
   * Instantiates a new Activation layer test base.
   *
   * @param layer the layer
   */
  public ActivationLayerTestBase(NNLayer layer) {
    this.layer = layer;
  }
  
  private final NNLayer layer;
  
  @Override
  public NNLayer getLayer() {
    return layer;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {2}
    };
  }
}