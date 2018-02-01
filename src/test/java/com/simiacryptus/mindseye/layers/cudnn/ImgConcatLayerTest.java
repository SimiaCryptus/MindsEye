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
import com.simiacryptus.mindseye.lang.cudnn.Precision;

import java.util.Random;

/**
 * The type Img concat layer apply.
 */
public abstract class ImgConcatLayerTest extends CuDNNLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Img concat layer apply.
   *
   * @param precision the precision
   */
  public ImgConcatLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {8, 8, 1}, {8, 8, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new ImgConcatLayer();
  }
  
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {200, 200, 3}, {200, 200, 3}
    };
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class;
  }
  
  /**
   * Test truncation feature that limits the image to N bands, discarding the last as needed.
   */
  public static class BandLimitTest extends ImgConcatLayerTest {
  
    /**
     * Instantiates a new Band limit apply.
     */
    public BandLimitTest() {
      super(Precision.Double);
    }
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 3}
      };
    }
  
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }
  
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(2);
    }
  }
  
  /**
   * Test truncation feature that both concatenates images and limits the image to N bands, discarding the last as
   * needed.
   */
  public static class BandConcatLimitTest extends ImgConcatLayerTest {
  
    /**
     * Instantiates a new Band limit apply.
     */
    public BandConcatLimitTest() {
      super(Precision.Double);
    }
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 2}, {1, 1, 2}
      };
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }
    
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(3);
    }
  }
  
  /**
   * Basic 64-bit apply
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
   * Basic 32-bit apply
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
