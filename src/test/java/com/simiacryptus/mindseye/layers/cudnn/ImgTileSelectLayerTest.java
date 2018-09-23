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
 * The type Img crop layer apply.
 */
public abstract class ImgTileSelectLayerTest extends CudaLayerTestBase {

  /**
   * Instantiates a new Img crop layer apply.
   */
  public ImgTileSelectLayerTest() {
    validateBatchExecution = false;
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {8, 8, 1}
    };
  }

  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{
        {1200, 1200, 1}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgTileSelectLayer(4, 4, 3, 3);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer.class;
  }

  /**
   * Basic Test
   */
  public static class Basic extends ImgTileSelectLayerTest {
  }

}
