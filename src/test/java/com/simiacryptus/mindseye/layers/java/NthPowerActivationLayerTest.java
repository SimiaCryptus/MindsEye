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

package com.simiacryptus.mindseye.layers.java;

/**
 * The type Nth power activation layer run.
 */
public class NthPowerActivationLayerTest {
  
  /**
   * The type Nth power run.
   */
  public static class NthPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Nth power run.
     */
    public NthPowerTest() {
      super(new NthPowerActivationLayer().setPower(2.5));
    }
  }
  
  /**
   * The type Square power run.
   */
  public static class SquarePowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Square power run.
     */
    public SquarePowerTest() {
      super(new NthPowerActivationLayer().setPower(2));
    }
  }
  
  /**
   * The type Sqrt power run.
   */
  public static class SqrtPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Sqrt power run.
     */
    public SqrtPowerTest() {
      super(new NthPowerActivationLayer().setPower(0.5));
    }
  }
  
  /**
   * The type Inv power run.
   */
  public static class InvPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv power run.
     */
    public InvPowerTest() {
      super(new NthPowerActivationLayer().setPower(-1));
    }
  }
  
  /**
   * The type Inv sqrt power run.
   */
  public static class InvSqrtPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Inv sqrt power run.
     */
    public InvSqrtPowerTest() {
      super(new NthPowerActivationLayer().setPower(-0.5));
    }
  }
  
  /**
   * The type Zero power run.
   */
  public static class ZeroPowerTest extends ActivationLayerTestBase {
    /**
     * Instantiates a new Zero power run.
     */
    public ZeroPowerTest() {
      super(new NthPowerActivationLayer().setPower(0));
    }
  }
  
}
