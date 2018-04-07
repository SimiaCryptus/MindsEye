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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.ComponentTestBase;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.Random;


/**
 * The type Img crop layer apply.
 */
public abstract class TileCycleLayerTest extends CudaLayerTestBase {
  
  /**
   * Instantiates a new Img crop layer apply.
   */
  public TileCycleLayerTest() {
    validateBatchExecution = false;
  }
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {1200, 1200, 3}
    };
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ImgCropLayer(5, 5);
  }
  
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
  }
  
  
  @Nullable
  @Override
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    @Nonnull ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setSamples(100).setBatches(10);
    return new ComponentTestBase<ToleranceStatistics>() {
      @Override
      public ToleranceStatistics test(@Nonnull NotebookOutput log, Layer component, Tensor... inputPrototype) {
        @Nullable PrintStream apiLog = null;
        try {
          apiLog = new PrintStream(log.file("cuda_perf.log"));
          CudaSystem.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          log.p(log.file((String) null, "cuda_perf.log", "GPU Log"));
          if (null != apiLog) {
            apiLog.close();
            CudaSystem.apiLog.remove(apiLog);
          }
        }
      }
      
      @Override
      protected void _free() {
        inner.freeRef();
        super._free();
      }
    };
  }
  
  /**
   * The type Chained.
   */
  public static class Chained extends TileCycleLayerTest {
  
    /**
     * Instantiates a new Chained.
     */
    public Chained() {
      validateDifferentials = false;
    }
    
    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull ImgCropLayer imgCropLayer = new ImgCropLayer(4, 5);
      //return wrap(imgCropLayer);
      return imgCropLayer;
    }
  
    /**
     * Wrap nn layer.
     *
     * @param imgCropLayer the img crop layer
     * @return the nn layer
     */
    @Nonnull
    public Layer wrap(ImgCropLayer imgCropLayer) {
      @Nonnull PipelineNetwork network = new PipelineNetwork();
      network.add(imgCropLayer);
      return network;
    }
    
    @Nonnull
    @Override
    public Class<? extends Layer> getTestClass() {
      return ImgCropLayer.class;
    }
    
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends TileCycleLayerTest {
  }
}
