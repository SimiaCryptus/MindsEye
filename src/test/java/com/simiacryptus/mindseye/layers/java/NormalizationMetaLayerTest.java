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
import com.simiacryptus.mindseye.layers.MetaLayerTestBase;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;

import java.util.Random;

/**
 * The type Normalization meta layer eval.
 */
public abstract class NormalizationMetaLayerTest extends MetaLayerTestBase {
  
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {3}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new NormalizationMetaLayer();
  }
  
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {10}
    };
  }
  
  @Override
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    return null;
    //return new BatchDerivativeTester(1e-2, 1e-5, 10);
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends NormalizationMetaLayerTest {
  }
  
}
