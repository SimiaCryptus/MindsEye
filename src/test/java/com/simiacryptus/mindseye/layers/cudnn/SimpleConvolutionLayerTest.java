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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

/**
 * The type Simple convolution layer test.
 */
public abstract class SimpleConvolutionLayerTest extends CudnnLayerTestBase {
  
  private final int bands;
  private final int radius;
  /**
   * The Layer.
   */
  SimpleConvolutionLayer layer;
  
  
  /**
   * Instantiates a new Simple convolution layer test.
   *
   * @param radius    the radius
   * @param bands     the bands
   * @param precision the precision
   */
  protected SimpleConvolutionLayerTest(final int radius, final int bands, final Precision precision) {
    this.radius = radius;
    this.bands = bands;
    layer = new SimpleConvolutionLayer(radius, radius, bands * bands).setPrecision(precision);
    layer.kernel.set(() -> random());
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {radius, radius, bands}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return layer;
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, bands}
    };
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    final ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, bands, bands, true);
    final Tensor tensor = new Tensor(layer.kernel.getDimensions());
    tensor.setByCoord(c -> {
      final int band = c.getCoords()[2];
      final int bandX = band % bands;
      final int bandY = (band - bandX) / bands;
      assert band == bandX + bandY * bands;
      final int bandT = bandY + bandX * bands;
      return layer.kernel.get(c.getCoords()[0], c.getCoords()[1], bandT);
    });
    convolutionLayer.kernel.set(tensor);
    return convolutionLayer;
  }
  
  /**
   * Maximally-basic single-value "convolution" in 64 bits
   */
  public static class Basic extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Basic() {
      super(1, 1, Precision.Double);
    }
  }
  
  /**
   * Typical 3x3 image convolution (64-bit)
   */
  public static class Image extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Image() {
      super(3, 3, Precision.Double);
    }
  }
  
  /**
   * Typical 3x3 image convolution (32-bit)
   */
  public static class Image_Float extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image float.
     */
    public Image_Float() {
      super(3, 3, Precision.Float);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  
  }
  
  /**
   * Basic single-band 3x3 image filter.
   */
  public static class Matrix extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Matrix.
     */
    public Matrix() {
      super(3, 1, Precision.Double);
    }
  }
  
  /**
   * Basic multi-band, 1-pixel-radius filter.
   */
  public static class MultiBand extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Multi band.
     */
    public MultiBand() {
      super(1, 3, Precision.Double);
    }
  }
  
}
