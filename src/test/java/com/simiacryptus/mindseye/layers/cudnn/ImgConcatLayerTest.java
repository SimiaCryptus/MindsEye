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

/**
 * The type Img concat layer run.
 */
public abstract class ImgConcatLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Img concat layer run.
   *
   * @param precision the precision
   */
  public ImgConcatLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {8, 8, 1}, {8, 8, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new ImgConcatLayer();
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {200, 200, 3}, {200, 200, 3}
    };
  }
  
  /**
   * Test truncation feature that limits the image to N bands, discarding the last as needed.
   */
  public static class BandLimitTest extends ImgConcatLayerTest {
  
    /**
     * Instantiates a new Band limit run.
     */
    public BandLimitTest() {
      super(Precision.Double);
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {2, 2, 2}, {2, 2, 2}
      };
    }
    
    @Override
    public NNLayer getLayer(final int[][] inputSize) {
      return new ImgConcatLayer().setMaxBands(3);
    }
  }
  
  /**
   * Basic 64-bit run
   */
  public static class Double extends ImgConcatLayerTest {
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
  public static class Float extends ImgConcatLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float);
    }
  }
  
}
