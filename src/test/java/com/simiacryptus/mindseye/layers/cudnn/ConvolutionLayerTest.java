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

/**
 * The type Convolution layer test.
 */
public abstract class ConvolutionLayerTest extends CudnnLayerTestBase {
  
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
   *
   * @param radius      the radius
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param precision   the precision
   */
  protected ConvolutionLayerTest(int radius, int inputBands, int outputBands, Precision precision) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision);
    convolutionLayer.kernel.fill(() -> random());
  }
  
  @Override
  public NNLayer getLayer(int[][] inputSize) {
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
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, inputBands}
    };
  }
  
  /**
   * The type Double.
   */
  public static class Double extends ConvolutionLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(1, 2, 2, Precision.Double);
    }
  }
  
  /**
   * The type Float.
   */
  public static class Float extends ConvolutionLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(1, 2, 2, Precision.Float);
    }
  }
  
  /**
   * The type Asymmetric test.
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
   * The type Irregular test float.
   */
  public static class IrregularTest_Float extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Irregular test float.
     */
    public IrregularTest_Float() {
      super(3, 7, 5, Precision.Float);
    }
  }
  
  /**
   * The type Irregular test.
   */
  public static class IrregularTest extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Irregular test.
     */
    public IrregularTest() {
      super(3, 7, 5, Precision.Double);
    }
  }
  
}
