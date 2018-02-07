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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.util.Util;

import java.util.Random;

/**
 * The type Entropy loss layer eval.
 */
public abstract class EntropyLossLayerTest extends LayerTestBase {
  
  @Override
  public SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(1e-4, 1e-8);
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {4}, {4}
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new EntropyLossLayer();
  }
  
  @Override
  public double random() {
    return Util.R.get().nextDouble();
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends EntropyLossLayerTest {
  }
  
}
