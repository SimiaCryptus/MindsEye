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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type Img eval layer apply.
 */
public abstract class ImgConcatLayerTest extends CudaLayerTestBase {

  private final Precision precision;
  private final int[] bandSeq;
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Img eval layer apply.
   *
   * @param precision     the precision
   * @param inputs        the inputs
   * @param bandsPerInput the bands per input
   * @param smallSize     the small size
   * @param largeSize     the large size
   */
  public ImgConcatLayerTest(final Precision precision, int inputs, int bandsPerInput, final int smallSize, final int largeSize) {
    this(precision, IntStream.range(0, inputs).map(i -> bandsPerInput).toArray(), smallSize, largeSize);
  }

  /**
   * Instantiates a new Img eval layer apply.
   *
   * @param precision the precision
   * @param bandSeq   the band seq
   * @param smallSize the small size
   * @param largeSize the large size
   */
  public ImgConcatLayerTest(final Precision precision, final int[] bandSeq, final int smallSize, final int largeSize) {
    this.bandSeq = bandSeq;
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return Arrays.stream(bandSeq).mapToObj(x -> new int[]{smallSize, smallSize, x}).toArray(i -> new int[i][]);
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgConcatLayer().setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return Arrays.stream(bandSeq).mapToObj(x -> new int[]{largeSize, largeSize, x}).toArray(i -> new int[i][]);
  }

  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class;
  }

  /**
   * Test truncation feature that limits the png to N bands, discarding the last as needed.
   */
  public static class BandLimitTest extends ImgConcatLayerTest {

    /**
     * Instantiates a new Band limit apply.
     */
    public BandLimitTest() {
      super(Precision.Double, 2, 1, 8, 1200);
    }

    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
          {1, 1, 3}
      };
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(2);
    }
  }

  /**
   * Test truncation feature that both concatenates images and limits the png to N bands, discarding the last as
   * needed.
   */
  public static class BandConcatLimitTest extends ImgConcatLayerTest {

    /**
     * Instantiates a new Band limit apply.
     */
    public BandConcatLimitTest() {
      super(Precision.Double, new int[]{2, 3, 4}, 2, 1200);
    }

    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new ImgConcatLayer().setMaxBands(8);
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
      super(Precision.Double, 2, 1, 8, 1200);
    }
  }
//
//  /**
//   * Basic 64-bit apply
//   */
//  public static class BigDouble extends Big {
//    /**
//     * Instantiates a new Double.
//     */
//    public BigDouble() {
//      super(Precision.Double, 8, 512);
//    }
//  }

  /**
   * The type BigTests.
   */
  public abstract static class Big extends ImgConcatLayerTest {

    /**
     * Instantiates a new BigTests.
     *
     * @param precision     the precision
     * @param inputs        the inputs
     * @param bandsPerInput the bands per input
     */
    public Big(final Precision precision, final int inputs, final int bandsPerInput) {
      super(precision, inputs, bandsPerInput, 8, 1200);
      this.validateDifferentials = false;
      setTestTraining(false);
    }

    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution) return null;
      return (new BatchingTester(1e-2) {
        @Override
        public double getRandom() {
          return random();
        }
      }).setBatchSize(5);
    }


    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
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
      super(Precision.Float, 2, 1, 8, 1200);
      tolerance = 1e-2;
    }
  }

}
