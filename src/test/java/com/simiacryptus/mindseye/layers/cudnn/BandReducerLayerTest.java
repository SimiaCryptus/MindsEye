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

import javax.annotation.Nonnull;
import java.util.Random;

/**
 * The type Img band bias layer apply.
 */
public abstract class BandReducerLayerTest extends CudaLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  private final double alpha;
  
  /**
   * Instantiates a new Img band bias layer apply.
   *
   * @param precision the precision
   * @param alpha     the alpha
   */
  public BandReducerLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
  }
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {2, 2, 1}
    };
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BandReducerLayer().setAlpha(alpha).setPrecision(precision);
  }
  
  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {32, 32, 3}
    };
  }
  
  /**
   * Basic apply in double (64-bit) precision
   */
  public static class Double extends BandReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 1.0);
    }
  }
  
  /**
   * Basic apply in double (64-bit) precision
   */
  public static class Negative extends BandReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Negative() {
      super(Precision.Double, -5.0);
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
      super(Precision.Double, 1.0);
    }
  
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {3, 5, 2}
      };
    }
  
    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
        {200, 100, 3}
      };
    }
    
  }
  
  /**
   * Basic apply using float (32-bit) precision.
   */
  public static class Float extends BandReducerLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 1.0);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
