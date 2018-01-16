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
import com.simiacryptus.mindseye.layers.cudnn.lang.CuDNN;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;
import java.util.Random;

/**
 * The type Fully connected layer apply.
 */
public abstract class FullyConnectedLayerTest extends CudnnLayerTestBase {
  
  private final int[] inputDim;
  private final FullyConnectedLayer fullyConnectedLayer;
  private final PipelineNetwork layer;
  
  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param dim the dim
   */
  public FullyConnectedLayerTest(int dim) {
    this(dim, dim);
  }
  
  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param inputDim  the input dim
   * @param outputDim the output dim
   */
  public FullyConnectedLayerTest(int inputDim, int outputDim) {
    this(new int[]{inputDim}, new int[]{outputDim});
  }
  
  /**
   * Instantiates a new Fully connected layer test.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedLayerTest(int[] inputDims, int[] outputDims) {
    this.inputDim = inputDims;
    this.fullyConnectedLayer = new FullyConnectedLayer(inputDims, outputDims).setWeightsLog(-2);
    this.layer = this.fullyConnectedLayer.explode();
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      inputDim
    };
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return FullyConnectedLayer.class;
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return layer;
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    Class<? extends NNLayer> referenceLayerClass = getReferenceLayerClass();
    return null == referenceLayerClass ? null : this.fullyConnectedLayer.as(referenceLayerClass);
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.FullyConnectedReferenceLayer.class;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(2);
    }
  }
  
  /**
   * Basic Test
   */
  public static class Big1 extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Big.
     */
    public Big1() {
      super(new int[]{2 * 1024}, new int[]{2 * 1024});
      validateDifferentials = false;
    }
  
    @Override
    public void run(NotebookOutput log) {
      String logName = "cuda_" + log.getName() + "_all.log";
      log.p(log.file((String) null, logName, "GPU Log"));
      CuDNN.addLog(new PrintStream(log.file(logName)));
      super.run(log);
    }
  
    @Override
    public Class<? extends NNLayer> getReferenceLayerClass() {
      return null;
    }
  
  }
  
  public static class Big2 extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Big.
     */
    public Big2() {
      super(new int[]{2048}, new int[]{2048});
      validateDifferentials = false;
    }
    
    @Override
    public Class<? extends NNLayer> getReferenceLayerClass() {
      return null;
    }
    
    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution) return null;
      return (new BatchingTester(1e-2) {
        @Override
        public double getRandom() {
          return random();
        }
      }).setBatchSize(2);
    }
    
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setBatches(1);
      return (log, component, inputPrototype) -> {
        PrintStream apiLog = null;
        try {
          String logName = "cuda_" + log.getName() + "_perf.log";
          log.p(log.file((String) null, logName, "GPU Log"));
          apiLog = new PrintStream(log.file(logName));
          CuDNN.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          if (null != apiLog) {
            apiLog.close();
            CuDNN.apiLog.remove(apiLog);
          }
        }
      };
    }
  }
  /**
   * Basic Test
   */
  public static class Big extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Big.
     */
    public Big() {
      super(new int[]{25088}, new int[]{4096});
      validateDifferentials = false;
    }
    
    @Override
    public Class<? extends NNLayer> getReferenceLayerClass() {
      return null;
    }
  
    @Override
    public ComponentTest<ToleranceStatistics> getBatchingTester() {
      if (!validateBatchExecution) return null;
      return (new BatchingTester(1e-2) {
        @Override
        public double getRandom() {
          return random();
        }
      }).setBatchSize(2);
    }
  
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setBatches(1);
      return (log, component, inputPrototype) -> {
        PrintStream apiLog = null;
        try {
          String logName = "cuda_" + log.getName() + "_perf.log";
          log.p(log.file((String) null, logName, "GPU Log"));
          apiLog = new PrintStream(log.file(logName));
          CuDNN.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          if (null != apiLog) {
            apiLog.close();
            CuDNN.apiLog.remove(apiLog);
          }
        }
      };
    }
  }
}
