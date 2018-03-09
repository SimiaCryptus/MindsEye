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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;

/**
 * The type Product layer run.
 */
public abstract class SquareActivationLayerTest extends CudaLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  private final double alpha;
  
  /**
   * Instantiates a new Product layer run.
   *
   * @param precision the precision
   * @param alpha
   */
  public SquareActivationLayerTest(final Precision precision, final double alpha) {
    this.precision = precision;
    this.alpha = alpha;
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
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new SquareActivationLayer().setPrecision(precision).setAlpha(alpha);
  }
  
  @Nullable
  @Override
  public Layer getReferenceLayer() {
    PipelineNetwork network = new PipelineNetwork();
    network.wrap(new LinearActivationLayer().setScale(alpha),
      network.wrap(new NthPowerActivationLayer().setPower(2), network.getInput(0)));
    return network;
    //return new NthPowerActivationLayer().setPower(2);
  }
  
  /**
   * Multiplication of 2 inputs using 64-bit precision
   */
  public static class Double extends SquareActivationLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 1.0);
    }
  }
  
  /**
   * Multiplication of 2 inputs using 64-bit precision
   */
  public static class Negative extends SquareActivationLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Negative() {
      super(Precision.Double, -1.0);
    }
  }
  
  /**
   * Multiplication of 2 inputs using 32-bit precision
   */
  public static class Float extends SquareActivationLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 1.0);
    }
    
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
    
  }
}
