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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;

/**
 * The type Product inputs layer run.
 */
public abstract class ProductInputsLayerTest extends LayerTestBase {
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new ProductInputsLayer();
  }
  
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester().setRandomizationMode(TrainingTester.RandomizationMode.Random);
  }
  
  /**
   * Multiply one multivariate input with a univariate input
   */
  public static class N1Test extends ProductInputsLayerTest {
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3}, {1}
      };
    }
  }
  
  /**
   * Multiply three multivariate inputs
   */
  public static class NNNTest extends ProductInputsLayerTest {
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3}, {3}, {3}
      };
    }
  }
  
  /**
   * Multiply two multivariate inputs
   */
  public static class NNTest extends ProductInputsLayerTest {
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {3}, {3}
      };
    }
  }
}
