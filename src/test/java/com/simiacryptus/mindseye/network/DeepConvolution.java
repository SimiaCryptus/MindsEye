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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.util.Util;

/**
 * The type Convolution network test.
 */
public abstract class DeepConvolution extends NLayerTest {
  
  /**
   * Instantiates a new Linear depth.
   *
   * @param dimList the dim list
   */
  public DeepConvolution(int[]... dimList) {
    super(dimList);
  }
  
  public void addLayer(PipelineNetwork network, int[] in, int[] out) {
    network.add(new ConvolutionLayer(3,3,in[2], out[2]).set(i->random()));
    network.add(new ImgBandBiasLayer(out[2]));
    network.add(getActivation());
  }
  
  public NNLayer getActivation() {
    return new ActivationLayer(ActivationLayer.Mode.RELU);
  }
  
  @Override
  public double random() {
    return 0.1 * Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 500.0;
  }
  
  @Override
  public int[] getInputDims() {
    return new int[]{5, 5, 3};
  }
  
  /**
   * The type Four layer.
   */
  public static class SigmoidPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public SigmoidPipeline() {
      super(
        new int[]{5,5,3},
        new int[]{5,5,3},
        new int[]{5,5,3}
      );
    }
  
    public NNLayer getActivation() {
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
      super(
        new int[]{3,3,3},
        new int[]{3,3,3},
        new int[]{3,3,3},
        new int[]{3,3,3}
      );
    }
    
  }
  
  /**
   * The type Four layer.
   */
  public static class ExpandPipeline extends DeepConvolution {
    /**
     * Instantiates a new Four layer.
     */
    public ExpandPipeline() {
      super(
        new int[]{5,5,9},
        new int[]{5,5,12},
        new int[]{5,5,48},
        new int[]{5,5,48}
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
      super(
        new int[]{4,4,2},
        new int[]{3,3,1},
        new int[]{2,2,1},
        new int[]{2,2,1}
      );
    }
    
  }
  
}
