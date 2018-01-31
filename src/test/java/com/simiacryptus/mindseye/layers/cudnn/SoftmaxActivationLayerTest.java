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

import com.simiacryptus.mindseye.lang.NNLayer;

import java.util.Random;

/**
 * The type Softmax activation layer run.
 */
public abstract class SoftmaxActivationLayerTest extends CuDNNLayerTestBase {
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{{4}};
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new SoftmaxActivationLayer();
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer.class;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends SoftmaxActivationLayerTest {
  }
}
