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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type Img concat layer run.
 */
public abstract class ImgConcatLayerTest extends CuDNNLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Inputs.
   */
  int inputs;
  /**
   * The Bands per input.
   */
  int bandsPerInput;
  
  /**
   * Instantiates a new Img concat layer run.
   *
   * @param precision     the precision
   * @param inputs        the inputs
   * @param bandsPerInput the bands per input
   */
  public ImgConcatLayerTest(final Precision precision, int inputs, int bandsPerInput) {
    this.precision = precision;
    this.inputs = inputs;
    this.bandsPerInput = bandsPerInput;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return IntStream.range(0, inputs).mapToObj(x -> new int[]{8, 8, bandsPerInput}).toArray(i -> new int[i][]);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new ImgConcatLayer();
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return IntStream.range(0, inputs).mapToObj(x -> new int[]{200, 200, bandsPerInput}).toArray(i -> new int[i][]);
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
     * Instantiates a new Band limit run.
     */
    public BandLimitTest() {
      super(Precision.Double, 2, 1);
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 3}
      };
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }
  
    @javax.annotation.Nonnull
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
     * Instantiates a new Band limit run.
     */
    public BandConcatLimitTest() {
      super(Precision.Double, 2, 1);
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 2}, {1, 1, 2}
      };
    }
  
    @javax.annotation.Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(new Random());
    }
  
    @javax.annotation.Nonnull
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
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
      super(Precision.Double, 2, 1);
    }
  }
  
  /**
   * Basic 64-bit run
   */
  public static class BigDouble extends Big {
    /**
     * Instantiates a new Double.
     */
    public BigDouble() {
      super(Precision.Double, 8, 512);
    }
  }
  
  /**
   * The type Big.
   */
  public static class Big extends ImgConcatLayerTest {
    /**
     * The Small size.
     */
    int smallSize;
    /**
     * The Lasrge size.
     */
    int lasrgeSize;
    
    /**
     * Instantiates a new Big.
     *
     * @param precision     the precision
     * @param inputs        the inputs
     * @param bandsPerInput the bands per input
     */
    public Big(final Precision precision, final int inputs, final int bandsPerInput) {
      super(precision, inputs, bandsPerInput);
      this.validateDifferentials = false;
      setTestTraining(false);
      this.lasrgeSize = 8;
      this.smallSize = 2;
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
    
    @Nonnull
    @Override
    public int[][] getSmallDims(final Random random) {
      return IntStream.range(0, inputs).mapToObj(x -> {
        return new int[]{smallSize, smallSize, bandsPerInput};
      }).toArray(i -> new int[i][]);
    }
    
    @Nonnull
    @Override
    public int[][] getLargeDims(final Random random) {
      return IntStream.range(0, inputs).mapToObj(x -> {
        return new int[]{lasrgeSize, lasrgeSize, bandsPerInput};
      }).toArray(i -> new int[i][]);
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
      super(Precision.Float, 2, 1);
    }
  }
  
}
