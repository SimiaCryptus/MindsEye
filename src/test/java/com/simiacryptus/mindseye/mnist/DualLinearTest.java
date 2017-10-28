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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer;
import com.simiacryptus.mindseye.layers.meta.NormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.PolynomialConvolutionNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The type Polynomial convolution test.
 */
public class DualLinearTest extends LinearTest {
  
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = null;
      network = new PipelineNetwork();
      network.add(new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}).setWeights(()->1e-8*(Math.random()-0.5)));
      network.add(new DenseSynapseLayer(new int[]{10}, new int[]{10}).setWeights(()->1e-8*(Math.random()-0.5)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
}
