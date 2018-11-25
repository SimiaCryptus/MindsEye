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

import com.simiacryptus.mindseye.lang.Layer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

/**
 * The type Rascaled subnet key apply.
 */
public abstract class ImgLinearSubnetLayerTest extends CudaLayerTestBase {

  private final Layer layer1 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final Layer layer2 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final Layer layer3 = new ActivationLayer(ActivationLayer.Mode.RELU);
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Img linear subnet key test.
   */
  public ImgLinearSubnetLayerTest() {
    testingBatchSize = 10;
    smallSize = 2;
    largeSize = 100;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {smallSize, smallSize, 3}
    };
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{
        {largeSize, largeSize, 3}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgLinearSubnetLayer()
        .add(0, 1, layer1)
        .add(1, 2, layer2)
        .add(2, 3, layer3);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  /**
   * Basic Test
   */
  public static class Basic extends ImgLinearSubnetLayerTest {

  }

}
