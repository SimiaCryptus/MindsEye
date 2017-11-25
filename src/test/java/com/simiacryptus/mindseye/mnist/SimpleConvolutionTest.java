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
import com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The type Simple convolution run.
 */
public class SimpleConvolutionTest extends LinearTest {
  
  
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    log.p("");
    return log.code(() -> {
      PipelineNetwork network = null;
      network = new PipelineNetwork();
      double carryOver = 0.5;
      int carryoverDenominator = 100;
      //network.fn(new CumNormalizationMetaLayer().setCarryOver(carryOver).setCarryoverDenominator(carryoverDenominator));
      network.add(new ConvolutionLayer(3, 3, 5).setWeights(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Avg));
//      network.fn(new CumNormalizationMetaLayer().setCarryOver(carryOver).setCarryoverDenominator(carryoverDenominator));
      network.add(new NormalizationMetaLayer());
      network.add(new FullyConnectedLayer(new int[]{13, 13, 5}, new int[]{10}).setWeights(() -> 1e-8 * (Math.random() - 0.5)));
//      network.fn(new CumNormalizationMetaLayer().setCarryOver(carryOver).setCarryoverDenominator(carryoverDenominator));
      network.add(new NormalizationMetaLayer());
      network.add(new LinearActivationLayer());
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
  @Override
  public NNLayer _test(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history) {
    log.p("This report trains a model using a recursive polynomial convolution layer.");
    DAGNetwork network = buildModel(log);
    run(log, monitoringRoot, monitor, trainingData, history, network);
    network.visitNodes(node -> {
      NNLayer layer = node.getLayer();
      if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer.class));
      }
      else if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer.class));
      }
    });
    run(log, monitoringRoot, monitor, trainingData, history, network);
    network.visitNodes(node -> {
      NNLayer layer = node.getLayer();
      if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer.class));
      }
      else if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.class));
      }
    });
    run(log, monitoringRoot, monitor, trainingData, history, network);
    testDataExpansion = data -> {
      Random random = new Random();
      return Stream.of(
        new Tensor[]{data[0], data[1]},
        new Tensor[]{addNoise(data[0]), data[1]},
        new Tensor[]{translate(random.nextInt(5) - 3, random.nextInt(5) - 3, data[0]), data[1]},
        new Tensor[]{translate(random.nextInt(5) - 3, random.nextInt(5) - 3, data[0]), data[1]}
      );
    };
    run(log, monitoringRoot, monitor, trainingData, history, network);
    return network;
  }
}
