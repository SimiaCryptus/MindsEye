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
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

/**
 * The type Convolution layer run.
 */
public abstract class ConvolutionLayerTest extends CudnnLayerTestBase {
  
  public static class Double extends ConvolutionLayerTest {
    public Double() {
      super(1, 2, 2, Precision.Double);
    }
  }
  
  public static class Float extends ConvolutionLayerTest {
    public Float() {
      super(1, 2, 2, Precision.Float);
    }
  }
  
  /**
   * The Radius.
   */
  final int radius;
  /**
   * The Input bands.
   */
  final int inputBands;
  /**
   * The Output bands.
   */
  final int outputBands;
  /**
   * The Convolution layer.
   */
  ConvolutionLayer convolutionLayer;
  
  /**
   * Instantiates a new Convolution layer test.
   *  @param radius      the radius
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param precision
   */
  protected ConvolutionLayerTest(int radius, int inputBands, int outputBands, Precision precision) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision);
    convolutionLayer.kernel.fill(() -> random());
  }
  
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
   * The type Asymmetric run.
   */
  public static class AsymmetricTest extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Asymmetric test.
     */
    public AsymmetricTest() {
      super(3, 3, 6, Precision.Double);
    }
    
  }
  
  /**
   * The type Irregular run.
   */
  public static class IrregularTest_Float extends ConvolutionLayerTest {
    
    /**
     * Instantiates a new Irregular test.
     */
    public IrregularTest_Float() {
      super(3, 7, 5, Precision.Float);
    }
  }
  
  /**
   * The type Irregular run.
   */
  public static class IrregularTest extends ConvolutionLayerTest {
    
    /**
     * Instantiates a new Irregular test.
     */
    public IrregularTest() {
      super(3, 7, 5, Precision.Double);
    }
  }
  
  /**
   * The type Asymmetric run.
   */
  public static class AsymmetricExplodedTest extends LayerTestBase {
  
    private Precision precision = Precision.Double;
  
    /**
     * Instantiates a new Asymmetric exploded test.
     */
    public AsymmetricExplodedTest() {
      super();
    }
    
    @Override
    public NNLayer getLayer() {
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new ImgConcatLayer().setMaxBands(3).setPrecision(precision),
        network.add(new SimpleConvolutionLayer(1, 1, 4).setWeights(this::random).setPrecision(precision), input),
        network.add(new SimpleConvolutionLayer(1, 1, 4).setWeights(this::random).setPrecision(precision), input));
      return network;
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{{1, 1, 2}};
    }
    
  }
  
}
