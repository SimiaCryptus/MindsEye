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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

/**
 * The type Product layer run.
 */
public abstract class ProductLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Product layer run.
   *
   * @param precision the precision
   */
  public ProductLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {2, 2, 1}, {2, 2, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new ProductLayer().setPrecision(precision);
  }
  
  /**
   * Multiplication of 2 inputs using 64-bit precision
   */
  public static class Double extends ProductLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double);
    }
  }
  
  /**
   * Multiplication of 3 inputs using 64-bit precision
   */
  public static class Double3 extends ProductLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double3() {
      super(Precision.Double);
    }
    
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {2, 2, 1}, {2, 2, 1}, {2, 2, 1}
      };
    }
  }
  
  /**
   * Multiplication of 2 inputs using 32-bit precision
   */
  public static class Float extends ProductLayerTest {
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
