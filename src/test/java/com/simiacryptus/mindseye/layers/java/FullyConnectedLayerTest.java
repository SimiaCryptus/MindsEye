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
import org.jetbrains.annotations.Nullable;

import java.util.Random;

/**
 * The type Fully connected layer eval.
 */
public abstract class FullyConnectedLayerTest extends LayerTestBase {
  
  @javax.annotation.Nonnull
  private final FullyConnectedLayer fullyConnectedLayer;
  private final int inputs;
  private final int outputs;
  
  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param inputs  the inputs
   * @param outputs the outputs
   */
  protected FullyConnectedLayerTest(int inputs, int outputs) {
    fullyConnectedLayer = new FullyConnectedLayer(new int[]{inputs}, new int[]{outputs});
    this.inputs = inputs;
    this.outputs = outputs;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {inputs}
    };
  }
  
  @Override
  public @Nullable Class<? extends NNLayer> getReferenceLayerClass() {
    return FullyConnectedReferenceLayer.class;
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    fullyConnectedLayer.addRef();
    return fullyConnectedLayer;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {super(3, 3);}
  }
  
  /**
   * The type Big.
   */
  public static class Big extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Big.
     */
    public Big() {
      super(25088, 4096);
      validateDifferentials = false;
    }
    
    @Override
    public Class<? extends NNLayer> getReferenceLayerClass() {
      return null;
    }
    
  }
  
  
}
