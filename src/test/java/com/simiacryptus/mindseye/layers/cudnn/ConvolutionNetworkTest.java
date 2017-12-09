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
import com.simiacryptus.mindseye.layers.DerivativeTester;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.java.ImgCropLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

/**
 * The type Convolution network test.
 */
public abstract class ConvolutionNetworkTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Convolution network test.
   *
   * @param precision the precision
   */
  public ConvolutionNetworkTest(Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public NNLayer getLayer() {
    
    PipelineNetwork network = new PipelineNetwork(2);
    network.add(new ConvolutionLayer(3, 3, 7, 3).setWeights(this::random).setPrecision(precision), network.getInput(1));
    network.add(new ImgBandBiasLayer(3).setPrecision(precision));
    network.add(new ActivationLayer(ActivationLayer.Mode.RELU).setPrecision(precision));
    network.add(new ImgCropLayer(4, 4));
    network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
      network.add(new MeanSqLossLayer(), network.getHead(), network.getInput(0))
    );
    return network;
    
    
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {4, 4, 3}, {5, 5, 7}
    };
  }
  
  /**
   * The type Double.
   */
  public static class Double extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }
  
  /**
   * The type Float.
   */
  public static class Float extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float);
    }
    
    @Override
    public DerivativeTester getDerivativeTester() {
      return new DerivativeTester(1e-2, 1e-3);
    }
    
  }
  
  /**
   * The type Debug.
   */
  public static class Debug extends LayerTestBase {
    
    @Override
    public NNLayer getLayer() {
      
      PipelineNetwork network = new PipelineNetwork(2);
//      network.add(new ConvolutionLayer(3, 3, 3, 3).setWeights(this::random), network.getInput(1));
//      network.add(new ImgBandBiasLayer(3));
//      network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      network.add(new ImgCropLayer(4, 4));
      network.add(new MeanSqLossLayer(), network.getHead(), network.getInput(0));
//      network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
//        network.add(new MeanSqLossLayer(), network.getHead(), network.getInput(0))
//      );
      return network;
      
      
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {4, 4, 3}, {5, 5, 3}
      };
    }
    
  }
  
  
}
