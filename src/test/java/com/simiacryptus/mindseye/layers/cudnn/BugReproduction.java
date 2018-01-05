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
import com.simiacryptus.mindseye.network.PipelineNetwork;

/**
 * Attempt to trigger software bug
 */
public class BugReproduction extends CudnnLayerTestBase {
  
  /**
   * Instantiates a new Irregular run float.
   */
  public BugReproduction() {
    this.validateDifferentials = false;
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    return null;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {1, 1, 512}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {16, 16, 512}
    };
  }
  
  @Override
  public NNLayer getLayer(int[][] inputSize) {
    PipelineNetwork model = new PipelineNetwork();
    model.add(new ImgZeroPaddingLayer(1, 1));
    model.add(new ConvolutionLayer(3, 3, 512, 512).setWeightsLog(-3).setPrecision(Precision.Double));
    model.add(new ImgBandBiasLayer(512));
    model.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    return model;
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return ConvolutionLayer.class;
  }
}
