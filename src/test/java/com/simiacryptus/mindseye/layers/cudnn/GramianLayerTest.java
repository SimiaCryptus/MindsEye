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
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.io.PrintStream;
import java.util.Random;

/**
 * The type Fully connected layer apply.
 */
public abstract class GramianLayerTest extends CudaLayerTestBase {
  
  /**
   * Instantiates a new Gramian layer test.
   */
  public GramianLayerTest() {
    testingBatchSize = 1;
  }
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {2, 2, 3}
    };
  }
  
  @Override
  public abstract int[][] getLargeDims(final Random random);
  
  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return GramianLayer.class;
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new GramianLayer();
  }
  
  @Override
  public Layer getReferenceLayer() {
    return null;
  }
  
  @Override
  public void run(NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }
  
  /**
   * Basic Test
   */
  public static class Image extends GramianLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Image() {
      super();
    }
  
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{
        {1000, 1000, 3}
      };
    }
  
  }
  
  public static class Deep extends GramianLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Deep() {
      super();
    }
    
    @Override
    public int[][] getLargeDims(final Random random) {
      return new int[][]{
        {100, 100, 512}
      };
    }
  }
  
  
}
