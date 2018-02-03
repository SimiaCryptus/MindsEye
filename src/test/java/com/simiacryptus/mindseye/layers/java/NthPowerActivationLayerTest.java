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

import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

/**
 * The type Nth power activation layer eval.
 */
public class NthPowerActivationLayerTest {
  
  /**
   * Tests x^-1 aka 1/x
   */
  public static class InvPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv power eval.
     */
    public InvPowerTest() {
      super(new NthPowerActivationLayer().setPower(-1));
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }
    
    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2) return random();
      return v;
    }
  }
  
  /**
   * Tests x^-1/2 aka 1/sqrt(x)
   */
  public static class InvSqrtPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv sqrt power eval.
     */
    public InvSqrtPowerTest() {
      super(new NthPowerActivationLayer().setPower(-0.5));
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-4);
    }
    
    @Override
    public double random() {
      final double v = super.random();
      if (Math.abs(v) < 0.2) return random();
      return v;
    }
  }
  
  /**
   * Tests an irregular power
   */
  public static class NthPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Nth power eval.
     */
    public NthPowerTest() {
      super(new NthPowerActivationLayer().setPower(Math.PI));
    }
  }
  
  /**
   * Tests x^1/2 aka sqrt(x)
   */
  public static class SqrtPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Sqrt power eval.
     */
    public SqrtPowerTest() {
      super(new NthPowerActivationLayer().setPower(0.5));
    }
  }
  
  /**
   * Tests x^2
   */
  public static class SquarePowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Square power eval.
     */
    public SquarePowerTest() {
      super(new NthPowerActivationLayer().setPower(2));
    }
  }
  
  /**
   * Tests x^0 aka 1
   */
  public static class ZeroPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Zero power eval.
     */
    public ZeroPowerTest() {
      super(new NthPowerActivationLayer().setPower(0));
    }
  }
  
}
