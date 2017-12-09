/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import com.simiacryptus.mindseye.layers.DerivativeTester;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;

/**
 * The type Activation layer re lu run.
 */
public abstract class ActivationLayerTest extends LayerTestBase {
  
  private Precision precision;
  
  public static class ReLu_Double extends ActivationLayerTest {
    public ReLu_Double() {
      super(ActivationLayer.Mode.RELU, Precision.Double);
    }
  }
  
  public static class ReLu_Float extends ActivationLayerTest {
    public ReLu_Float() {
      super(ActivationLayer.Mode.RELU, Precision.Float);
    }
  }
  
  public static class Sigmoid_Double extends ActivationLayerTest {
    public Sigmoid_Double() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Double);
    }
  }
  
  public static class Sigmoid_Float extends ActivationLayerTest {
    public Sigmoid_Float() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Float);
    }
  }
  
  final ActivationLayer.Mode mode;
  
  public ActivationLayerTest(ActivationLayer.Mode mode, Precision precision) {
    this.mode = mode;
    this.precision = precision;
  }
  
  @Override
  public NNLayer getLayer() {
    return new ActivationLayer(mode).setPrecision(precision);
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{{1, 1, 3}};
  }
  
  @Override
  public DerivativeTester getDerivativeTester() {
    return new DerivativeTester(1e-2, 1e-4);
  }
}
