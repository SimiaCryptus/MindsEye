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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type BinarySumLayerTest layer apply.
 */
public abstract class SumInputsLayerTest extends CudaLayerTestBase {
  
  private static int largeSize;
  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Input bands.
   */
  int inputBands;
  /**
   * The Inputs.
   */
  int inputs;
  
  /**
   * Instantiates a new Product layer apply.
   *
   * @param precision  the precision
   * @param inputBands the input bands
   * @param inputs     the inputs
   * @param largeSize  the large size
   */
  public SumInputsLayerTest(final Precision precision, int inputBands, int inputs, final int largeSize) {
    this.precision = precision;
    this.inputBands = inputBands;
    this.inputs = inputs;
    SumInputsLayerTest.largeSize = largeSize;
  }
  
  @Override
  public int[][] getSmallDims(Random random) {
    return IntStream.range(0, inputs).mapToObj(i -> new int[]{2, 2, inputBands}).toArray(i -> new int[i][]);
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer().setPrecision(precision);
  }
  
  @Override
  public int[][] getLargeDims(Random random) {
    return IntStream.range(0, inputs).mapToObj(i -> new int[]{largeSize, largeSize, inputBands}).toArray(i -> new int[i][]);
  }
  
  @Override
  public Class<? extends Layer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.SumInputsLayer.class;
  }
  
  /**
   * The type Double list.
   */
  public static class Double_List extends SumInputsLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_List() {
      super(Precision.Double, 1, 5, 1200);
    }
    
  }
  
  /**
   * Ensures addition can be used to implement a doubling (x2) function
   */
  public static class OnePlusOne extends CudaLayerTestBase {
  
  
    /**
     * Instantiates a new Asymmetric apply.
     */
    public OnePlusOne() {
      super();
    }
  
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return com.simiacryptus.mindseye.layers.java.SumInputsLayer.class;
    }
  
    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.wrap(new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer(), input, input);
      return network;
    }
  
    @Override
    protected Class<?> getTargetClass() {
      return SumInputsLayer.class;
    }
  
    @Override
    public Layer getReferenceLayer() {
      @Nonnull PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.wrap(new SumInputsLayer(), input, input);
      return network;
    }
  
    @Nonnull
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {4, 4, 1}
      };
    }
  
    @Nonnull
    @Override
    public int[][] getLargeDims(Random random) {
      return new int[][]{
        {1200, 800, 1}
      };
    }
  
  }
  
  /**
   * Adds using double (64-bit) precision, C = A + B
   */
  public static class Big_Double_Add extends Big {
    /**
     * Instantiates a new Double.
     */
    public Big_Double_Add() {
      super(Precision.Double, 256, 8, 100);
      testingBatchSize = 2;
    }
  }
  
  /**
   * The type Double add.
   */
  public static class Double_Add extends SumInputsLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_Add() {
      super(Precision.Double, 1, 2, 1200);
    }
  }
  
  
  /**
   * Adds using float (32-bit) precision, C = A + B
   */
  public static class Float_Add extends SumInputsLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float_Add() {
      super(Precision.Float, 1, 2, 1200);
    }
    
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
    
  }
  
  /**
   * Basic Test
   */
  public abstract static class Big extends SumInputsLayerTest {
  
    /**
     * Instantiates a new BigTests.
     *
     * @param precision  the precision
     * @param inputBands the input bands
     * @param inputs     the inputs
     * @param largeSize  the large size
     */
    public Big(final Precision precision, int inputBands, int inputs, final int largeSize) {
      super(precision, inputBands, inputs, largeSize);
      validateDifferentials = false;
      setTestTraining(false);
      testingBatchSize = 5;
    }
    
    @Override
    public Class<? extends Layer> getReferenceLayerClass() {
      return null;
    }
    
    @Nullable
    @Override
    protected ComponentTest<ToleranceStatistics> getJsonTester() {
      logger.warn("Disabled Json Test");
      return null;
    }
    
    @Nullable
    @Override
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      logger.warn("Disabled Performance Test");
      return null;
    }
  
  }
  
}
