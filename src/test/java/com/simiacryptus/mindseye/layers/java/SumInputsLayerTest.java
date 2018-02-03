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
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.Random;

/**
 * The type Sum inputs layer eval.
 */
public class SumInputsLayerTest {
  /**
   * The type N 1 eval.
   */
  public static class N1Test extends LayerTestBase {
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {3}, {1}
      };
    }
  
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new SumInputsLayer();
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
        {100}, {1}
      };
    }
    
  }
  
  /**
   * The type Nn eval.
   */
  public static class NNTest extends LayerTestBase {
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {3}, {3}
      };
    }
  
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new SumInputsLayer();
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
        {100}, {100}
      };
    }
    
  }
  
  /**
   * Ensures addition can be used to implement a doubling (x2) function
   */
  public static class OnePlusOne extends LayerTestBase {
  
    /**
     * Instantiates a new Asymmetric run.
     */
    public OnePlusOne() {
      super();
    }
    
    
    @Override
    public NNLayer getLayer(int[][] inputSize, Random random) {
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new SumInputsLayer(), input, input);
      return network;
    }
    
    @Override
    public NNLayer getReferenceLayer() {
      return null;
    }
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 1}
      };
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(random);
    }
    
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }
    
  }
}
