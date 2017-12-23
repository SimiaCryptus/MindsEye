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
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

/**
 * The type Weight extractor test.
 */
public class WeightExtractorTest extends LayerTestBase {
  
  /**
   * Instantiates a new Weight extractor test.
   */
  public WeightExtractorTest() {
    validateDifferentials = false;
    validateBatchExecution = false;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {0}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    PipelineNetwork network = new PipelineNetwork();
    FullyConnectedLayer layer = new FullyConnectedLayer(new int[]{3}, new int[]{3});
    network.add(layer);
    network.add(new WeightExtractor(0, layer), new DAGNode[]{});
    return network;
  }
}
