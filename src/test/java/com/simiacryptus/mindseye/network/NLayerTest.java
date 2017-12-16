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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;

import java.util.Arrays;
import java.util.List;

/**
 * The type N layer test.
 */
public abstract class NLayerTest extends LayerTestBase {
  
  /**
   * The Dim list.
   */
  final List<int[]> dimList;
  
  /**
   * Instantiates a new N layer test.
   *
   * @param dimList the dim list
   */
  public NLayerTest(int[]... dimList) {
    this.dimList = Arrays.asList(dimList);
  }
  
  @Override
  public NNLayer getLayer(int[][] inputSize) {
    PipelineNetwork network = new PipelineNetwork(1);
    int[] last = inputSize[0];
    for (int[] dims : dimList) {
      addLayer(network, last, dims);
      last = dims;
    }
    return network;
  }
  
  /**
   * Add layer.
   *
   * @param network the network
   * @param in      the in
   * @param out    the dims
   */
  public abstract void addLayer(PipelineNetwork network, int[] in, int[] out);
  
}
