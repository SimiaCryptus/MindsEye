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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;

/**
 * The type Mean sq loss layer test.
 */
public abstract class MeanSqLossLayerTest extends LayerTestBase {
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {3, 3, 1}, {3, 3, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new MeanSqLossLayer();
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    return new com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer();
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, 3}, {100, 100, 3}
    };
  }
  
  /**
   * Basic test.
   */
  public class Basic extends MeanSqLossLayerTest {
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3, 3, 2}, {3, 3, 2}
      };
    }
  }
  
  /**
   * Test using asymmetric input.
   */
  public class Asymetric extends MeanSqLossLayerTest {
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {2, 3, 1}, {2, 3, 1}
      };
    }
  }
  
}
