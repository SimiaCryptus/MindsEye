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
import com.simiacryptus.mindseye.test.DerivativeTester;

/**
 * The type Product layer test.
 */
public abstract class BinarySumLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Product layer test.
   *
   * @param precision the precision
   */
  public BinarySumLayerTest(Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public NNLayer getLayer() {
    return new BinarySumLayer().setPrecision(precision);
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {2,2,1}, {2,2,1}
    };
  }
  
  /**
   * The type Double.
   */
  public static class Double_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_Add() {
      super(Precision.Double);
    }
  }
  
  /**
   * The type Double.
   */
  public static class Double_Subtract extends BinarySumLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_Subtract() {
      super(Precision.Double);
    }
    
    @Override
    public NNLayer getLayer() {
      return new BinarySumLayer(1.0, -1.0).setPrecision(precision);
    }
  
  }
  
  /**
   * The type Float.
   */
  public static class Float_Avg extends BinarySumLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float_Avg() {
      super(Precision.Float);
    }
    
    @Override
    public DerivativeTester getDerivativeTester() {
      return new DerivativeTester(1e-2, 1e-3);
    }
  
    @Override
    public NNLayer getLayer() {
      return new BinarySumLayer(0.5, 0.5).setPrecision(precision);
    }
  
  }
  
  /**
   * The type Float.
   */
  public static class Float_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float_Add() {
      super(Precision.Float);
    }
    
    @Override
    public DerivativeTester getDerivativeTester() {
      return new DerivativeTester(1e-2, 1e-3);
    }
    
  }
}
