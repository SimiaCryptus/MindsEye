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
import javax.annotation.Nullable;
import java.util.Random;

/**
 * The type Img band bias layer apply.
 */
public abstract class BandAvgReducerLayerTest extends CudaLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;
  private final double alpha;
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Img band bias layer apply.
   *
   * @param precision the precision
   * @param alpha     the alphaList
   * @param smallSize the small size
   * @param largeSize the large size
   */
  public BandAvgReducerLayerTest(final Precision precision, final double alpha, final int smallSize, final int largeSize) {
    this.precision = precision;
    this.alpha = alpha;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {smallSize, smallSize, 1}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BandAvgReducerLayer().setAlpha(alpha).setPrecision(precision);
  }

  @Nullable
  @Override
  public Layer getReferenceLayer() {
    return new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg).setAlpha(alpha).setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
        {largeSize, largeSize, 3}
    };
  }

  /**
   * Basic apply in double (64-bit) precision
   */
  public static class Double extends BandAvgReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 1.0, 8, 1200);
    }
  }

  /**
   * Basic apply in double (64-bit) precision
   */
  public static class Negative extends BandAvgReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Negative() {
      super(Precision.Double, -5.0, 8, 1200);
    }
  }

  /**
   * Inputs asymmetric (height != width) images
   */
  public static class Asymmetric extends BandAvgReducerLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Asymmetric() {
      super(Precision.Double, 1.0, 4, 1200);
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
          {1200, 800, 3}
      };
    }

  }

  /**
   * Basic apply using float (32-bit) precision.
   */
  public static class Float extends BandAvgReducerLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 1.0, 4, 1200);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  }
}
