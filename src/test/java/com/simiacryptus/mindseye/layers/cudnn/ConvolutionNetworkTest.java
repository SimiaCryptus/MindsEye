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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

/**
 * The type Convolution network run.
 */
public abstract class ConvolutionNetworkTest extends CudnnLayerTestBase {
  
  private final int radius;
  private final int inputBands;
  private final int outputBands;
  /**
   * The Precision.
   */
  final Precision precision;
  private final NNLayer layer;
  
  /**
   * Instantiates a new Convolution layer run.
   *
   * @param radius      the radius
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param precision   the precision
   */
  protected ConvolutionNetworkTest(final int radius, final int inputBands, final int outputBands, final Precision precision) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision);
    convolutionLayer.getKernel().set(() -> random());
    this.precision = precision;
    this.layer = convolutionLayer.explode();
  }
  
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {4, 4, 3}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return this.layer;
    
    
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (64-bit)
   */
  public static class DoubleConvolutionNetwork extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Double.
     */
    public DoubleConvolutionNetwork() {
      super(3, 4, 8, Precision.Double);
    }
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (32-bit)
   */
  public static class FloatConvolutionNetwork extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Float.
     */
    public FloatConvolutionNetwork() {
      super(3, 4, 8, Precision.Float);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  
  }
  
  
}
