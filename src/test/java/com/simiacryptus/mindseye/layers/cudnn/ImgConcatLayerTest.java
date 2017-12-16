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
 * The type Img concat layer test.
 */
public abstract class ImgConcatLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Img concat layer test.
   *
   * @param precision the precision
   */
  public ImgConcatLayerTest(Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public NNLayer getLayer() {
    return new ImgConcatLayer();
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {2, 2, 1}, {2, 2, 1}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, 1}, {100, 100, 1}
    };
  }
  
  /**
   * The type Double.
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
   * The type Float.
   */
  public static class Float extends ImgConcatLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float);
    }
  }
  
  /**
   * The type Band limit test.
   */
  public static class BandLimitTest extends ImgConcatLayerTest {

    /**
     * Instantiates a new Band limit test.
     */
    public BandLimitTest() {
      super(Precision.Double);
    }
    
    @Override
    public NNLayer getLayer() {
      return new ImgConcatLayer().setMaxBands(3);
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {2, 2, 2}, {2, 2, 2}
      };
    }
  }
  
}
