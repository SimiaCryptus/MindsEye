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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import java.util.Random;

/**
 * The type Rascaled subnet layer apply.
 */
public abstract class StochasticSamplingSubnetLayerTest extends LayerTestBase {
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {6, 6, 1}
    };
  }
  
  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    PipelineNetwork subnetwork = new PipelineNetwork(1);
    subnetwork.wrap(new ProductLayer(),
      subnetwork.getInput(0),
      subnetwork.add(new StochasticBinaryNoiseLayer(0.5, 1.0, inputSize[0]), new DAGNode[]{}));
    
    StochasticSamplingSubnetLayer tileSubnetLayer = new StochasticSamplingSubnetLayer(subnetwork, 2);
    subnetwork.freeRef();
    return tileSubnetLayer;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends StochasticSamplingSubnetLayerTest {
  }
  
}
