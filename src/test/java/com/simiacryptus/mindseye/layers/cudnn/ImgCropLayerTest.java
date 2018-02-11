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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.GpuSystem;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.ComponentTestBase;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.Random;


/**
 * The type Img crop layer run.
 */
public abstract class ImgCropLayerTest extends CuDNNLayerTestBase {
  
  /**
   * Instantiates a new Img crop layer run.
   */
  public ImgCropLayerTest() {
    validateBatchExecution = false;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new ImgCropLayer(5, 5);
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
  }
  
  
  @Nullable
  @Override
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    @javax.annotation.Nonnull ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setSamples(100).setBatches(10);
    return new ComponentTestBase<ToleranceStatistics>() {
      @Override
      public ToleranceStatistics test(@javax.annotation.Nonnull NotebookOutput log, NNLayer component, Tensor... inputPrototype) {
        @Nullable PrintStream apiLog = null;
        try {
          apiLog = new PrintStream(log.file("cuda_perf.log"));
          GpuSystem.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          log.p(log.file((String) null, "cuda_perf.log", "GPU Log"));
          if (null != apiLog) {
            apiLog.close();
            GpuSystem.apiLog.remove(apiLog);
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
  public static class Chained extends ImgCropLayerTest {
  
    /**
     * Instantiates a new Chained.
     */
    public Chained() {
      validateDifferentials = false;
    }
  
    @javax.annotation.Nonnull
    @Override
    public NNLayer getLayer(int[][] inputSize, Random random) {
      @javax.annotation.Nonnull ImgCropLayer imgCropLayer = new ImgCropLayer(4, 5);
      //return wrap(imgCropLayer);
      return imgCropLayer;
    }
  
    /**
     * Wrap nn layer.
     *
     * @param imgCropLayer the img crop layer
     * @return the nn layer
     */
    @javax.annotation.Nonnull
    public NNLayer wrap(ImgCropLayer imgCropLayer) {
      @javax.annotation.Nonnull PipelineNetwork network = new PipelineNetwork();
      network.add(imgCropLayer);
      return network;
    }
  
    @javax.annotation.Nonnull
    @Override
    public Class<? extends NNLayer> getTestClass() {
      return ImgCropLayer.class;
    }
    
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends ImgCropLayerTest {
  }
}
