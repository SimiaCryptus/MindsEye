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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import java.util.Random;

/**
 * The type Img band bias layer run.
 */
public abstract class BandReducerLayerTest extends CudaLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Img band bias layer run.
   *
   * @param precision the precision
   */
  public BandReducerLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BandReducerLayer().setPrecision(precision);
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  /**
   * Basic run in double (64-bit) precision
   */
  public static class Double extends BandReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }
  
  /**
   * Inputs asymmetric (height != width) images
   */
  public static class Asymmetric extends BandReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Asymmetric() {
      super(Precision.Double);
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {3, 5, 2}
      };
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
        {100, 60, 3}
      };
    }
    
  }
  
  /**
   * Basic run using float (32-bit) precision.
   */
  public static class Float extends BandReducerLayerTest {
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
