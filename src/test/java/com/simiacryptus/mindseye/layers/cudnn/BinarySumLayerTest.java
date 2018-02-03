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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * The type BinarySumLayerTest layer run.
 */
public abstract class BinarySumLayerTest extends CuDNNLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  
  /**
   * Instantiates a new Product layer run.
   *
   * @param precision the precision
   */
  public BinarySumLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {8, 8, 1}, {8, 8, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new BinarySumLayer().setPrecision(precision);
  }
  
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {200, 200, 3}, {200, 200, 3}
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
      return IntStream.range(0, 5).mapToObj(i -> new int[]{2, 2, 2}).toArray(i -> new int[i][]);
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return IntStream.range(0, 5).mapToObj(i -> new int[]{200, 200, 3}).toArray(i -> new int[i][]);
    }
    
  }
  
  /**
   * Ensures addition can be used to implement a doubling (x2) function
   */
  public static class OnePlusOne extends CuDNNLayerTestBase {
  
    /**
     * Instantiates a new Asymmetric run.
     */
    public OnePlusOne() {
      super();
    }
    
    
    @Override
    public NNLayer getLayer(int[][] inputSize, Random random) {
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new BinarySumLayer(), input, input);
      return network;
    }
    
    @Override
    public NNLayer getReferenceLayer() {
      PipelineNetwork network = new PipelineNetwork();
      DAGNode input = network.getInput(0);
      network.add(new SumInputsLayer(), input, input);
      return network;
    }
    
    @Override
    public int[][] getSmallDims(Random random) {
      return new int[][]{
        {1, 1, 1}
      };
    }
    
    @Override
    public int[][] getLargeDims(Random random) {
      return getSmallDims(random);
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
  
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
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
  
    @Override
    public NNLayer getLayer(final int[][] inputSize, Random random) {
      return new BinarySumLayer(0.5, 0.5).setPrecision(precision);
    }
  
  }
}
