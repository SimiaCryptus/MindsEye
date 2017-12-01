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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

/**
 * The type Convolution layer run.
 */
public class ConvolutionLayerTest extends LayerTestBase {
  final int radius;
  final int inputBands;
  final int outputBands;
  
  public ConvolutionLayerTest() {
    this(1,2,2);
  }
  
  protected ConvolutionLayerTest(int radius, int inputBands, int outputBands) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands);
    convolutionLayer.kernel.fill(() -> random());
  }
  
  /**
   * The Convolution layer.
   */
  ConvolutionLayer convolutionLayer;
  
  @Override
  public NNLayer getLayer() {
    return convolutionLayer;
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer referenceLayer = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(radius, radius, inputBands, outputBands, true);
    referenceLayer.kernel.set(convolutionLayer.kernel);
    return referenceLayer;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {5, 5, inputBands}
    };
  }
  
  /**
   * The type Asymmetric test.
   */
  public static class AsymmetricTest extends ConvolutionLayerTest {
  
    public AsymmetricTest() {
      super(3,7,3);
    }
  
  }
  
  /**
   * The type Irregular test.
   */
  public static class IrregularTest extends ConvolutionLayerTest {
    
    public IrregularTest() {
      super(3,3,6);
    }
  }
  
  /**
   * The type Asymmetric test.
   */
  public static class AsymmetricExplodedTest extends LayerTestBase {
    
    public AsymmetricExplodedTest() {
      super();
    }
    
    @Override
    public NNLayer getLayer() {
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new ImgConcatLayer().setMaxBands(3),
        network.add(new SimpleConvolutionLayer(1,1,4).setWeights(this::random), input),
        network.add(new SimpleConvolutionLayer(1,1,4).setWeights(this::random), input));
      return network;
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{ { 1,1,2 } };
    }
    
  }
  
}
