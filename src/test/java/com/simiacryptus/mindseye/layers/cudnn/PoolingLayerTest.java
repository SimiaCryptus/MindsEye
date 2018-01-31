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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import java.util.Random;

/**
 * The type Pooling layer apply.
 */
public abstract class PoolingLayerTest extends CuDNNLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Pooling layer apply.
   *
   * @param precision the precision
   */
  public PoolingLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new PoolingLayer().setPrecision(precision);
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @Override
  public int[][] getPerfDims(Random random) {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  /**
   * Basic 64-bit apply.
   */
  public static class Double extends PoolingLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }
  
  /**
   * Test using an asymmetric window size.
   */
  public static class Asymmetric extends PoolingLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Asymmetric() {
      super(Precision.Double);
    }
    
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new PoolingLayer().setPrecision(precision).setWindowY(4);
    }
    
  }
  
  /**
   * Basic 32-bit apply.
   */
  public static class Float extends PoolingLayerTest {
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
