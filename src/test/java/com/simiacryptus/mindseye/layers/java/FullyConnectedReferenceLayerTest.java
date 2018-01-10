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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;

import java.util.Random;

/**
 * The type Fully connected layer run.
 */
public abstract class FullyConnectedReferenceLayerTest extends LayerTestBase {
  private final int[] outputDims;
  private final int[] inputDims;
  private final FullyConnectedReferenceLayer layer;
  
  
  /**
   * Instantiates a new Fully connected reference layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedReferenceLayerTest(int[] inputDims, int[] outputDims) {
    this.outputDims = outputDims;
    this.inputDims = inputDims;
    this.layer = new FullyConnectedReferenceLayer(getInputDims(new Random())[0], outputDims).set(i -> random());
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      inputDims
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return layer;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends FullyConnectedReferenceLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(new int[]{2}, new int[]{2});
    }
  }
  
  /**
   * Basic Test
   */
  public static class Image extends FullyConnectedReferenceLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Image() {
      super(new int[]{3, 3, 3}, new int[]{2, 2, 4});
    }
  }
  
}
