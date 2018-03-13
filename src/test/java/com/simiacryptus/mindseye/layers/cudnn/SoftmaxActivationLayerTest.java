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
import java.util.Random;

/**
 * The type Softmax activation layer apply.
 */
public abstract class SoftmaxActivationLayerTest extends CudaLayerTestBase {
  
  private final SoftmaxActivationLayer.SoftmaxAlgorithm algorithm;
  private final SoftmaxActivationLayer.SoftmaxMode mode;
  
  /**
   * Instantiates a new Softmax activation layer test.
   *
   * @param algorithm the algorithm
   * @param mode      the mode
   */
  public SoftmaxActivationLayerTest(final SoftmaxActivationLayer.SoftmaxAlgorithm algorithm, final SoftmaxActivationLayer.SoftmaxMode mode) {
    this.algorithm = algorithm;
    this.mode = mode;
  }
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 2, 3}};
  }
  
  @Override
  public int[][] getLargeDims(final Random random) {
    return new int[][]{{200, 200, 3}};
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new SoftmaxActivationLayer().setMode(mode).setAlgorithm(algorithm);
  }
  
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
    //return com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer.class;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends SoftmaxActivationLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.INSTANCE);}
  }
  
  /**
   * Basic Test
   */
  public static class Pixel extends SoftmaxActivationLayerTest {
    /**
     * Instantiates a new Pixel.
     */
    public Pixel() {super(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);}
  }
  
  /**
   * The type Pixel log.
   */
  public static class PixelLog extends SoftmaxActivationLayerTest {
    /**
     * Instantiates a new Pixel log.
     */
    public PixelLog() {super(SoftmaxActivationLayer.SoftmaxAlgorithm.LOG, SoftmaxActivationLayer.SoftmaxMode.CHANNEL);}
  }
}
