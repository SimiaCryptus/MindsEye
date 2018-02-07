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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;

/**
 * The type Convolution network eval.
 */
public abstract class DeepLinear extends NLayerTest {
  
  /**
   * Instantiates a new Linear depth.
   *
   * @param dimList the dim list
   */
  public DeepLinear(final int[]... dimList) {
    super(dimList);
  }
  
  @Override
  public void addLayer(final @NotNull PipelineNetwork network, final @NotNull int[] in, final @NotNull int[] dims) {
    network.add(new FullyConnectedLayer(in, dims).set(this::random));
    network.add(new BiasLayer(dims));
    network.add(getActivation());
  }
  
  /**
   * Gets activation.
   *
   * @return the activation
   */
  public @NotNull NNLayer getActivation() {
    return new ReLuActivationLayer();
  }
  
  @Override
  public @NotNull int[] getInputDims() {
    return new int[]{5, 5, 3};
  }
  
  @Override
  public double random() {
    return 0.1 * Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 500.0;
  }
  
  /**
   * The type Four layer.
   */
  public static class NarrowingPipeline extends DeepLinear {
    /**
     * Instantiates a new Four layer.
     */
    public NarrowingPipeline() {
      super(
        new int[]{4, 4, 2},
        new int[]{3, 3, 1},
        new int[]{2, 2, 1},
        new int[]{2, 2, 1}
           );
    }
    
  }
  
  /**
   * The type Four layer.
   */
  public static class SigmoidPipeline extends DeepLinear {
    /**
     * Instantiates a new Four layer.
     */
    public SigmoidPipeline() {
      super(
        new int[]{10},
        new int[]{10},
        new int[]{10},
        new int[]{10}
           );
    }
  
    @Override
    public @NotNull NNLayer getActivation() {
      return new SigmoidActivationLayer();
    }
  
  }
  
  /**
   * The type Four layer.
   */
  public static class UniformPipeline extends DeepLinear {
    /**
     * Instantiates a new Four layer.
     */
    public UniformPipeline() {
      super(
        new int[]{10},
        new int[]{10},
        new int[]{10},
        new int[]{10}
           );
    }
  
  }
  
}
