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
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;

import java.io.PrintStream;


/**
 * The type Img crop layer run.
 */
public abstract class ImgCropLayerTest extends CudnnLayerTestBase {
  
  /**
   * Instantiates a new Img crop layer run.
   */
  public ImgCropLayerTest() {
    validateBatchExecution = false;
    ImgCropLayer.disable = false;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {200, 200, 3}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new ImgCropLayer(5, 5);
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.ImgCropLayer.class;
  }
  
  
  @Override
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setSamples(100);
    return (log1, component, inputPrototype) -> {
      try {
        CuDNN.apiLog = new PrintStream(log1.file("cuda_perf.log"));
        return inner.test(log1, component, inputPrototype);
      } finally {
        log1.p(log1.file((String) null, "cuda_perf.log", "GPU Log"));
        CuDNN.apiLog.close();
        CuDNN.apiLog = null;
      }
    };
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends ImgCropLayerTest {
  }
}
