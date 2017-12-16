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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.util.Util;

/**
 * The type Convolution network test.
 */
public abstract class LinearDepth extends NLayerTest {
  
  /**
   * Instantiates a new Linear depth.
   *
   * @param dimList the dim list
   */
  public LinearDepth(int[]... dimList) {
    super(dimList);
  }
  
  public void addLayer(PipelineNetwork network, int[] in, int[] dims) {
    network.add(new FullyConnectedLayer(in, dims).setWeights(this::random));
    network.add(new BiasLayer(dims));
    network.add(new ReLuActivationLayer());
  }
  
  @Override
  public double random() {
    return 0.1 * Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 500.0;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {5, 5, 3}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, 3}
    };
  }
  
  /**
   * The type One layer.
   */
  public static class OneLayer extends LinearDepth {
    /**
     * Instantiates a new One layer.
     */
    public OneLayer() {
      super(
        new int[]{5}
      );
    }
    
  }
  
  /**
   * The type Two layer.
   */
  public static class TwoLayer extends LinearDepth {
    /**
     * Instantiates a new Two layer.
     */
    public TwoLayer() {
      super(
        new int[]{5},
        new int[]{5}
      );
    }
    
  }
  
  /**
   * The type Three layer.
   */
  public static class ThreeLayer extends LinearDepth {
    /**
     * Instantiates a new Three layer.
     */
    public ThreeLayer() {
      super(
        new int[]{5},
        new int[]{5},
        new int[]{5}
      );
    }
    
  }
  
  /**
   * The type Four layer.
   */
  public static class FourLayer extends LinearDepth {
    /**
     * Instantiates a new Four layer.
     */
    public FourLayer() {
      super(
        new int[]{5},
        new int[]{5},
        new int[]{5},
        new int[]{5}
      );
    }
    
  }
  
}
