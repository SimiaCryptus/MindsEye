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

package com.simiacryptus.mindseye.mnist;

import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.java.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.util.io.NotebookOutput;

public class SimpleNormalizedTest extends LinearTest {
  
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    log.p("");
    return log.code(() -> {
      PipelineNetwork network = null;
      network = new PipelineNetwork();
      network.add(new NormalizationMetaLayer());
      network.add(new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}).setWeights(()->1e-8*(Math.random()-0.5)));
      network.add(new NormalizationMetaLayer());
      network.add(new LinearActivationLayer());
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
}
