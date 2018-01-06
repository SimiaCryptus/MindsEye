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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.NNLayer;

/**
 * The type Fully connected layer run.
 */
public abstract class FullyConnectedLayerTest extends CudnnLayerTestBase {
  
  private final int inputDim;
  private final int outputDim;
  
  public FullyConnectedLayerTest(int dim) {
    this.inputDim = dim;
    this.outputDim = dim;
  }
  
  public FullyConnectedLayerTest(int inputDim, int outputDim) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {inputDim}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new FullyConnectedLayer(new int[]{inputDim}, new int[]{outputDim}).setWeightsLog(-2);
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.FullyConnectedLayer.class;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends FullyConnectedLayerTest {
    public Basic() {
      super(8);
    }
  }
  
  /**
   * Basic Test
   */
  public static class Big extends FullyConnectedLayerTest {
    public Big() {
      super(128);
      validateDifferentials = false;
    }
  }
}
