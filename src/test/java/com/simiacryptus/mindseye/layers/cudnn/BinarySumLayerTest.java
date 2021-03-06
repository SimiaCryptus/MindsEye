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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import javax.annotation.Nonnull;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type BinarySumLayerTest key apply.
 */
public abstract class BinarySumLayerTest extends CudaLayerTestBase {

  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Large size.
   */
  final int largeSize;
  /**
   * The Small size.
   */
  int smallSize;

  /**
   * Instantiates a new Product key apply.
   *
   * @param precision the precision
   */
  public BinarySumLayerTest(final Precision precision) {
    this.precision = precision;
    smallSize = 2;
    largeSize = 1200;
  }

  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
        {smallSize, smallSize, 1}, {smallSize, smallSize, 1}
    };
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new BinarySumLayer().setPrecision(precision);
  }

  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
        {largeSize, largeSize, 1}, {largeSize, largeSize, 1}
    };
  }

  /**
   * The type Double list.
   */
  public static class Double_List extends BinarySumLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_List() {
      super(Precision.Double);
    }

    @Override
    public int[][] getSmallDims(Random random) {
      return IntStream.range(0, 5).mapToObj(i -> new int[]{smallSize, smallSize, 2}).toArray(i -> new int[i][]);
    }

    @Override
    public int[][] getLargeDims(Random random) {
      return IntStream.range(0, 5).mapToObj(i -> new int[]{largeSize, largeSize, 3}).toArray(i -> new int[i][]);
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


    @Nonnull
    @Override
    public Layer getLayer(int[][] inputSize, Random random) {
      @Nonnull PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.wrap(new BinarySumLayer(), input, input).freeRef();
      return network;
    }

    @Override
    public Layer getReferenceLayer() {
      @Nonnull PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.wrap(new SumInputsLayer(), input, input).freeRef();
      return network;
    }

    @Override
    protected Class<?> getTargetClass() {
      return BinarySumLayer.class;
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
  public static class Double_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_Add() {
      super(Precision.Double);
    }
  }

  /**
   * Subtracts using double (64-bit) precision, C = A - B
   */
  public static class Double_Subtract extends BinarySumLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double_Subtract() {
      super(Precision.Double);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new BinarySumLayer(1.0, -1.0).setPrecision(precision);
    }

  }

  /**
   * Adds using float (32-bit) precision, C = A + B
   */
  public static class Float_Add extends BinarySumLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float_Add() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

  }

  /**
   * Binary averaging using float (32-bit) precision, C = (A + B) / 2
   */
  public static class Float_Avg extends BinarySumLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float_Avg() {
      super(Precision.Float);
    }

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }

    @Nonnull
    @Override
    public Layer getLayer(final int[][] inputSize, Random random) {
      return new BinarySumLayer(0.5, 0.5).setPrecision(precision);
    }

  }
}
