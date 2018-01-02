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
 * The type Img band bias layer run.
 */
public abstract class ImgBandBiasLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Img band bias layer run.
   *
   * @param precision the precision
   */
  public ImgBandBiasLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {8, 8, 3}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new ImgBandBiasLayer(3).setPrecision(precision).addWeights(this::random);
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  /**
   * Basic 64-bit run
   */
  public static class Double extends ImgBandBiasLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }
  
  /**
   * Basic 32-bit run
   */
  public static class Float extends ImgBandBiasLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
