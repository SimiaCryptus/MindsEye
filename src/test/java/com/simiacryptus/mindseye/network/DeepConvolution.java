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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;

/**
 * The type Convolution network apply.
 */
public abstract class DeepConvolution extends NLayerTest {
  
  /**
   * The Height.
   */
  static int height = 64;
  /**
   * The Width.
   */
  static int width = 64;
  /**
   * The Radius.
   */
  int radius;
  
  /**
   * Instantiates a new Linear depth.
   *
   * @param radius  the radius
   * @param dimList the length list
   */
  public DeepConvolution(int radius, final int[]... dimList) {
    super(dimList);
    this.radius = radius;

  }
  
  
  @Override
  public void addLayer(@Nonnull final PipelineNetwork network, final int[] in, final int[] out) {
    assert in[0] == out[0];
    assert in[1] == out[1];
    network.add(new ConvolutionLayer(radius, radius, in[2], out[2]).set(i -> random()));
    network.add(new ImgBandBiasLayer(out[2]));
    network.add(getActivation());
  }
  
  /**
   * Gets activation.
   *
   * @return the activation
   */
  @Nonnull
  public Layer getActivation() {
    return new ActivationLayer(ActivationLayer.Mode.RELU);
  }
  
  @Nonnull
  @Override
  public int[] getInputDims() {
    return new int[]{width, height, 3};
  }
  
  @Override
  public double random() {
    return 0.1 * Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 500.0;
  }
  
  /**
   * The type Four layer.
   */
  public static class ExpandPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public ExpandPipeline() {
      super(3,
        new int[]{width, height, 9},
        new int[]{width, height, 12},
        new int[]{width, height, 48},
        new int[]{width, height, 48}
      );
    }
    
  }
  
  /**
   * The type Four layer.
   */
  public static class NarrowingPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public NarrowingPipeline() {
      super(3,
        new int[]{width, height, 2},
        new int[]{width, height, 1},
        new int[]{width, height, 1},
        new int[]{width, height, 1}
      );
    }
    
  }
  
  /**
   * The type Four layer.
   */
  public static class SigmoidPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public SigmoidPipeline() {
      super(3,
        new int[]{width, height, 5},
        new int[]{width, height, 5},
        new int[]{width, height, 5},
        new int[]{width, height, 5},
        new int[]{width, height, 5}
      );
    }
  
    @Nonnull
    @Override
    public Layer getActivation() {
      return new ActivationLayer(ActivationLayer.Mode.SIGMOID);
    }
  
  }
  
  /**
   * The type Four layer.
   */
  public static class UniformPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public UniformPipeline() {
      super(3,
        new int[]{width, height, 3},
        new int[]{width, height, 3},
        new int[]{width, height, 3},
        new int[]{width, height, 3}
      );
    }
  
  }
  
}
