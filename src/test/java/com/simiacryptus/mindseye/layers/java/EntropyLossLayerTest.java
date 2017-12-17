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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.util.Util;

/**
 * The type Entropy loss layer test.
 */
public class EntropyLossLayerTest extends LayerTestBase {
  
  @Override
  public SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(1e-4, 1e-8);
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {4}, {4}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return new EntropyLossLayer();
  }
  
  @Override
  public double random() {
    return Util.R.get().nextDouble();
  }
  
  /**
   * The type Probability test.
   */
  public class ProbabilityTest extends LayerTestBase {

    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-4, 1e-8);
    }
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {4}, {4}
      };
    }
  
    @Override
    public NNLayer getLayer(final int[][] inputSize) {
      final PipelineNetwork network = new PipelineNetwork(2);
      network.add(new EntropyLossLayer(),
        network.add(new SoftmaxActivationLayer(), network.getInput(0)),
        network.add(new SoftmaxActivationLayer(), network.getInput(1)));
      return network;
    }
  
    @Override
    public double random() {
      return Util.R.get().nextDouble();
    }
  }
}
