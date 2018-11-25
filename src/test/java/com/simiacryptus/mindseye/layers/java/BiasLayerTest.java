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

import javax.annotation.Nonnull;
import java.util.Random;

/**
 * The type Bias key apply.
 */
public abstract class BiasLayerTest extends LayerTestBase {

  private final int dimension;

  /**
   * Instantiates a new Bias key apply.
   *
   * @param dimension the dimension
   */
  public BiasLayerTest(int dimension) {
    this.dimension = dimension;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {dimension}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BiasLayer(dimension).addWeights(this::random);
  }

  /**
   * Basic Test
   */
  public static class Basic extends BiasLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(5);
    }
  }

  /**
   * Tests applying a single bias value on all inputs.
   */
  public static class Reducing extends BiasLayerTest {

    /**
     * Instantiates a new Reducing.
     */
    public Reducing() {
      super(5);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new BiasLayer(1).addWeights(this::random);
    }

  }

}
