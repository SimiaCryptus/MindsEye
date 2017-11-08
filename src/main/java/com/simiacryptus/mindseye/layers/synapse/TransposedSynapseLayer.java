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

package com.simiacryptus.mindseye.layers.synapse;

import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Tensor;

/**
 * The type Transposed synapse layer.
 */
public class TransposedSynapseLayer extends MappedSynapseLayer {
  
  private final DenseSynapseLayer sibling;
  
  /**
   * Instantiates a new Transposed synapse layer.
   */
  protected TransposedSynapseLayer() {
    super();
    sibling = null;
  }
  
  /**
   * Instantiates a new Transposed synapse layer.
   *
   * @param sibling the sibling
   */
  public TransposedSynapseLayer(final DenseSynapseLayer sibling) {
    super(sibling.outputDims, sibling.inputDims);
    this.sibling = sibling;
  }
  
  @Override
  public int getMappedIndex(Coordinate inputCoord, Coordinate outputCoord) {
    return inputCoord.index + Tensor.dim(inputDims) * outputCoord.index;
  }
  
  @Override
  public Tensor buildWeights() {
    Tensor weights = sibling.getWeights();
    int[] dims = weights.getDimensions();
    assert (2 == dims.length);
    double[] data = weights.getData();
    return new Tensor(data, new int[]{dims[1], dims[0]});
  }
  
}
