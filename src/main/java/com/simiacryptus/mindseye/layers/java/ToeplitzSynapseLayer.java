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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.Util;

/**
 * The type Toeplitz synapse layer.
 */
public class ToeplitzSynapseLayer extends MappedSynapseLayer {
  
  private final int radius;
  
  /**
   * Instantiates a new Toeplitz synapse layer.
   */
  protected ToeplitzSynapseLayer() {
    super();
    this.radius = 0;
  }
  
  /**
   * Instantiates a new Toeplitz synapse layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public ToeplitzSynapseLayer(final int[] inputDims, final int[] outputDims) {
    this(inputDims, outputDims, Integer.MAX_VALUE);
  }
  
  /**
   * Instantiates a new Toeplitz synapse layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   * @param radius     the radius
   */
  public ToeplitzSynapseLayer(final int[] inputDims, final int[] outputDims, int radius) {
    super(inputDims, outputDims);
    this.radius = radius;
  }
  
  @Override
  public int getMappedIndex(Coordinate inputCoord, Coordinate outputCoord) {
    int[] coordVector = new int[inputDims.length];
    int[] spareVector = new int[outputDims.length - inputDims.length];
    for (int i = 0; i < inputDims.length; i++) {
      coordVector[i] = inputCoord.coords[i] - outputCoord.coords[i] + (outputDims[i] - 1);
    }
    for (int i = 0; i < spareVector.length; i++) {
      spareVector[i] = outputCoord.coords[i + inputDims.length];
    }
    return allowVector(coordVector) ? getWeights().index(concat(coordVector, spareVector)) : -1;
  }
  
  @Override
  public Tensor buildWeights() {
    assert (this.inputDims.length <= this.outputDims.length);
    final int inputs = Tensor.dim(this.inputDims);
    final int outs = Tensor.dim(this.outputDims);
    int[] weightDims = new int[this.outputDims.length];
    for (int i = 0; i < weightDims.length; i++)
      weightDims[i] = (i < this.inputDims.length ? this.inputDims[i] : 1) + this.outputDims[i] - 1;
    Tensor tensor = new Tensor(weightDims);
    double[] tensorData = tensor.getData();
    for (int i = 0; i < tensorData.length; i++) {
      double ratio = Math.sqrt(6. / (inputs + outs));
      double fate = Util.R.get().nextDouble();
      tensorData[i] = (1 - 2 * fate) * ratio;
    }
    return tensor;
  }
  
  private boolean allowVector(int[] coordVector) {
    double total = 0;
    for (int i = 0; i < coordVector.length; i++) total += coordVector[i];
    return total < radius;
  }
  
}
