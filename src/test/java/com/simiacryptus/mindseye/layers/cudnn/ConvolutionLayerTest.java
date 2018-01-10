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
import com.simiacryptus.mindseye.lang.Tensor;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * The type Convolution layer run.
 */
public abstract class ConvolutionLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Input bands.
   */
  final int inputBands;
  /**
   * The Output bands.
   */
  final int outputBands;
  /**
   * The Radius.
   */
  final int radius;
  /**
   * The Convolution layer.
   */
  ConvolutionLayer convolutionLayer;
  
  /**
   * Instantiates a new Convolution layer run.
   *
   * @param radius      the radius
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param precision   the precision
   * @param batchBands
   */
  protected ConvolutionLayerTest(final int radius, final int inputBands, final int outputBands, final Precision precision, int batchBands) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision).setBatchBands(batchBands);
    convolutionLayer.getKernel().set(() -> random());
  }
  
  @Test
  public void verifyWeights() {
    ExplodedConvolutionGrid explodedNetwork = this.convolutionLayer.getExplodedNetwork();
    int[] kernelDims = this.convolutionLayer.getKernel().getDimensions();
    Tensor testData = new Tensor(kernelDims).map(x -> Math.random());
    explodedNetwork.write(testData);
    Tensor echo = explodedNetwork.extractKernel();
    Assert.assertEquals(testData, echo);
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      {5, 5, inputBands}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return convolutionLayer;
  }
  
  @Override
  public int[][] getPerfDims(Random random) {
    return new int[][]{
      {100, 100, inputBands}
    };
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer.class;
  }
  
  /**
   * Increases the number of color bands from 3 to 6 (radius 3; 64-bit precision)
   */
  public static class BandExpand extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Asymmetric run.
     */
    public BandExpand() {
      super(3, 3, 6, Precision.Double, 16);
    }
    
  }
  
  /**
   * Increases the number of color bands from 3 to 6 (radius 3; 64-bit precision)
   */
  public static class SqGrid extends ConvolutionLayerTest {
    
    /**
     * Instantiates a new Asymmetric run.
     */
    public SqGrid() {
      super(3, 4, 4, Precision.Double, 2);
    }
    
  }
  
  /**
   * Increases the number of color bands from 3 to 6 (radius 3; 64-bit precision)
   */
  public static class IrregularGrid extends ConvolutionLayerTest {
    
    /**
     * Instantiates a new Asymmetric run.
     */
    public IrregularGrid() {
      super(3, 5, 3, Precision.Double, 2);
    }
    
  }
  
  /**
   * Reduces the number of color bands from 6 to 3 (radius 3; 64-bit precision)
   */
  public static class BandReduceTest extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Asymmetric run.
     */
    public BandReduceTest() {
      super(3, 6, 3, Precision.Double, 16);
    }
    
  }
  
  /**
   * Test using 64-bit precision with a radius of 1
   */
  public static class Double extends ConvolutionLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(3, 2, 3, Precision.Double, 16);
    }
  }
  
  /**
   * Tests with no zero-padding; the output will be radius-1 smaller than the input. This currently tests a workaround
   * where CuDNN does not seem to support convolutions that change resolution.
   */
  public static class NoPadding extends ConvolutionLayerTest {
    /**
     * Instantiates a new Double.
     */
    public NoPadding() {
      super(3, 3, 3, Precision.Double, 16);
      convolutionLayer.setPaddingXY(0, 0);
    }
  
    @Override
    public Class<? extends NNLayer> getReferenceLayerClass() {
      // BUG: Reference aparapi implementation does not seem to implement nonstandard padding correctly
      return null;
    }
  
    @Override
    public NNLayer getReferenceLayer() {
      return super.getReferenceLayer();
    }
  }
  
  /**
   * Test using 32-bit precision with a radius of 1
   */
  public static class Float extends ConvolutionLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(1, 2, 2, Precision.Float, 16);
    }
  }
  
  /**
   * Convert from 7 bands to 5; this is meant to not divide evenly for testing. (64-bit)
   */
  public static class IrregularTest extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Irregular run.
     */
    public IrregularTest() {
      super(3, 7, 5, Precision.Double, 16);
    }
  }
  
  /**
   * Convert from 7 bands to 5; this is meant to not divide evenly for testing. (32-bit)
   */
  public static class IrregularTest_Float extends ConvolutionLayerTest {
  
    /**
     * Instantiates a new Irregular run float.
     */
    public IrregularTest_Float() {
      super(3, 7, 5, Precision.Float, 16);
    }
  }
  
}
