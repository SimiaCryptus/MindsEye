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
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.*;

import java.util.ArrayList;

/**
 * The type Conv pipeline eval.
 */
public class ConvPipelineTest extends PipelineTest {
  
  /**
   * Instantiates a new Conv pipeline eval.
   */
  public ConvPipelineTest() {
    super(
      buildList_1()
    );
  }
  
  /**
   * Build list 1 nn layer [ ].
   *
   * @return the nn layer [ ]
   */
  public static NNLayer[] buildList_1() {
    @javax.annotation.Nonnull final ArrayList<NNLayer> network = new ArrayList<NNLayer>();
    
    network.add(new ConvolutionLayer(3, 3, 3, 10).set(i -> 1e-8 * (Math.random() - 0.5)));
    network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    network.add(new ReLuActivationLayer());
    network.add(new ImgCropLayer(126, 126));
    
    network.add(new ConvolutionLayer(3, 3, 10, 20).set(i -> 1e-8 * (Math.random() - 0.5)));
    network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    network.add(new ReLuActivationLayer());
    network.add(new ImgCropLayer(62, 62));
    
    network.add(new ConvolutionLayer(5, 5, 20, 30).set(i -> 1e-8 * (Math.random() - 0.5)));
    network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
    network.add(new ReLuActivationLayer());
    network.add(new ImgCropLayer(18, 18));
    
    network.add(new ConvolutionLayer(3, 3, 30, 40).set(i -> 1e-8 * (Math.random() - 0.5)));
    network.add(new PoolingLayer().setWindowX(4).setWindowY(4).setMode(PoolingLayer.PoolingMode.Avg));
    network.add(new ReLuActivationLayer());
    network.add(new ImgCropLayer(4, 4));
    
    network.add(new ImgBandBiasLayer(40));
    network.add(new FullyConnectedLayer(new int[]{4, 4, 40}, new int[]{100}).set(() -> 0.001 * (Math.random() - 0.45)));
    network.add(new SoftmaxActivationLayer());
    
    return network.toArray(new NNLayer[]{});
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[] getInputDims() {
    return new int[]{256, 256, 3};
  }
}
