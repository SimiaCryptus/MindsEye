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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.mnist.MnistTestBase;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The type Qqn test.
 */
public class QQNTest extends MnistTestBase {
  
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new LinearActivationLayer().setScale(1.0 / 1000));
      network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{100}));
      network.add(new SinewaveActivationLayer());
      network.add(new FullyConnectedLayer(new int[]{100}, new int[]{10}));
      network.add(new LinearActivationLayer());
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
  @Override
  public void train(NotebookOutput log, NNLayer network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      //return new IterativeTrainer(new StochasticArrayTrainable(trainingData, supervisedNetwork, 10000))
      return new ValidatingTrainer(
        new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000, 10000),
        new ArrayTrainable(trainingData, supervisedNetwork)
      )
        .setMonitor(monitor)
        //.setOrientation(new ValidatingOrientationStrategy(new QQN()))
        .setOrientation(new QQN())
        //.setLineSearchFactory(name->name.contains("QQN") ? new QuadraticSearch().setCurrentRate(1.0) : new QuadraticSearch().setCurrentRate(1e-6))
        .setTimeout(30, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
    });
  }
  
}
