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

package com.simiacryptus.mindseye.layers.cudnn.conv;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.CudaLayerTestBase;
import com.simiacryptus.mindseye.layers.java.FullyConnectedReferenceLayer;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.BatchingTester;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

/**
 * The type Fully connected layer apply.
 */
public abstract class FullyConnectedLayerTest extends CudaLayerTestBase {

  /**
   * The Input dim.
   */
  @Nonnull
  protected final int[] inputDim;
  /**
   * The Fully connected layer.
   */
  @Nonnull
  protected final FullyConnectedLayer fullyConnectedLayer;
  /**
   * The LayerBase.
   */
  @Nonnull
  protected final Layer layer;

  /**
   * Instantiates a new Fully connected layer allocationOverflow.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   * @param batchBands the batch bands
   */
  public FullyConnectedLayerTest(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
    this.inputDim = inputDims;
    this.fullyConnectedLayer = new FullyConnectedLayer(inputDims, outputDims).setWeightsLog(-2);
    this.layer = this.fullyConnectedLayer.setBatchBands(batchBands).explode();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        inputDim
    };
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return FullyConnectedLayer.class;
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    layer.addRef();
    return layer;
  }

  @Override
  public Layer getReferenceLayer() {
    @Nullable Class<? extends Layer> referenceLayerClass = getReferenceLayerClass();
    return null == referenceLayerClass ? null : this.fullyConnectedLayer.as(referenceLayerClass);
  }

  @Nullable
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return FullyConnectedReferenceLayer.class;
  }

  @Override
  public void run(NotebookOutput log) {
//    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    CudaSystem.addLog(new PrintStream(log.file(logName)));
    super.run(log);
  }

  /**
   * Basic Test
   */
  public static class Basic extends FullyConnectedLayerTest {
    /**
     * Instantiates a new Basic.
     */
    public Basic() {
      super(new int[]{2}, new int[]{2}, 512);
    }
  }

  /**
   * Basic Test
   */
  public abstract static class BigTests extends FullyConnectedLayerTest {

    /**
     * Instantiates a new BigTests.
     *
     * @param inputDims  the input dims
     * @param outputDims the output dims
     * @param batchBands the batch bands
     */
    public BigTests(@Nonnull int[] inputDims, @Nonnull int[] outputDims, int batchBands) {
      super(inputDims, outputDims, batchBands);
      validateDifferentials = false;
      setTestTraining(false);
    }

    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
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
      }).setBatchSize(5);
    }

    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
      //return super.getJsonTester();
    }

    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
      //return super.getPerformanceTester();
    }
  }

  /**
   * Large-dimension test using the size of the largest layer in VGG16
   */
  public static class Big_VGG extends BigTests {
    /**
     * Instantiates a new BigTests.
     */
    public Big_VGG() {
      super(new int[]{25088}, new int[]{4096}, 25088 / 2);
    }

  }

  /**
   * Large-dimension test
   */
  public static class Big1 extends BigTests {
    /**
     * Instantiates a new BigTests.
     */
    public Big1() {
      super(new int[]{2 * 1024}, new int[]{2 * 1024}, 512);
    }

  }


}
