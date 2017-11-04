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
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SinewaveActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.mnist.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class LBFGSTest extends MnistTestBase {
  
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new LinearActivationLayer().setScale(1.0/1000));
      network.add(new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{100}));
      network.add(new SinewaveActivationLayer());
      network.add(new DenseSynapseLayer(new int[]{100}, new int[]{10}));
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
        new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000,10000),
        new ArrayTrainable(trainingData, supervisedNetwork).cached()
      )
               .setMonitor(monitor)
               //.setOrientation(new ValidatingOrientationStrategy(new LBFGS()))
               .setOrientation(new LBFGS())
               .setLineSearchFactory(name -> name.contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0) : new QuadraticSearch())
               .setTimeout(30, TimeUnit.MINUTES)
               .setMaxIterations(500)
               .run();
    });
  }
  
}
