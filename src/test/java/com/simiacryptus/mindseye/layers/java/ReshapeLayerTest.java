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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import javax.annotation.Nullable;
import java.util.Random;

/**
 * The type Rascaled subnet layer eval.
 */
public abstract class ReshapeLayerTest extends LayerTestBase {
  
  private final int[] outputDims;
  private final int[] inputDims;
  
  /**
   * Instantiates a new Reshape layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  protected ReshapeLayerTest(int[] inputDims, int[] outputDims) {
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      inputDims
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ReshapeLayer(outputDims);
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends ReshapeLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {super(new int[]{6, 6, 1}, new int[]{1, 1, 36});}
  }
  
  /**
   * The type Basic 1.
   */
  public static class Basic1 extends ReshapeLayerTest {
    /**
     * Instantiates a new Basic 1.
     */
    public Basic1() {super(new int[]{1, 1, 32}, new int[]{1, 1, 32});}
  }
  
  /**
   * The type BigTests 0.
   */
  public static class Big0 extends Big {
    /**
     * Instantiates a new BigTests 0.
     */
    public Big0() {super(256);}
  
  }
  
  /**
   * The type BigTests 1.
   */
  public static class Big1 extends Big {
    /**
     * Instantiates a new BigTests 1.
     */
    public Big1() {super(new int[]{4, 4, 256}, new int[]{1, 1, 2048});}
  }
  
  /**
   * The type BigTests 2.
   */
  public static class Big2 extends Big {
    /**
     * Instantiates a new BigTests 2.
     */
    public Big2() {super(new int[]{1, 1, 2048}, new int[]{4, 4, 256});}
  }
  
  /**
   * The type BigTests.
   */
  public abstract static class Big extends ReshapeLayerTest {
  
    /**
     * Instantiates a new Big.
     *
     * @param size the size
     */
    public Big(int size) {this(new int[]{1, 1, size}, new int[]{1, 1, size});}
  
    /**
     * Instantiates a new BigTests.
     *
     * @param inputDims  the input dims
     * @param outputDims the output dims
     */
    public Big(int[] inputDims, int[] outputDims) {
      super(inputDims, outputDims);
      validateDifferentials = false;
    }
    
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
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
      //return super.getJsonTester();
    }
    
    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
      //return super.getPerformanceTester();
    }
  }
}
