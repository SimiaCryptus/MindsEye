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

import java.util.Random;

/**
 * The type Max dropout noise layer run.
 */
public abstract class MaxDropoutNoiseLayerTest extends ActivationLayerTestBase {
  /**
   * Instantiates a new Max dropout noise layer run.
   */
  public MaxDropoutNoiseLayerTest() {
    super(new MaxDropoutNoiseLayer(2, 2, 1));
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends MaxDropoutNoiseLayerTest {
  }
  
}
