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

import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.trainable.DeltaHoldoverArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class PolynomialNetworkTest extends MnistTestBase {
  
  PolynomialNetwork tree;
  int trainingSize = 10000;
  int iterationsPerSample = 25;
  int maxIterations = 500;
  int timeoutMinutes = 5;
  
  @Override
  public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      Trainable trainable = new DeltaHoldoverArrayTrainable(trainingData, supervisedNetwork, trainingSize);
      return new IterativeTrainer(trainable)
               .setIterationsPerSample(iterationsPerSample)
               .setMonitor(monitor)
               .setOrientation(new QQN())
               //.setLineSearchFactory(name->name.contains("QQN") ? new QuadraticSearch().setCurrentRate(1.0) : new QuadraticSearch().setCurrentRate(1e-6))
               .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
               .setMaxIterations(maxIterations)
               .run();
    });
    
  }
  
  @Override
  public PipelineNetwork _test(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history) {
    log.p("First, define a model:");
    PipelineNetwork network = buildModel(log);
    addMonitoring(network, monitoringRoot);
  
    iterationsPerSample = 5;
    trainingSize = 5000;
    timeoutMinutes = 5;
    run(log, monitoringRoot, monitor, trainingData, history, network);

//    iterationsPerSample = 0;
//    trainingSize = 0;
    trainingSize = 10000;
    timeoutMinutes = 5;
    run(log, monitoringRoot, monitor, trainingData, history, network);
  
    timeoutMinutes = 120;
    tree.addTerm(1);
    addMonitoring(network, monitoringRoot);
    run(log, monitoringRoot, monitor, trainingData, history, network);
  
    timeoutMinutes = 120;
    tree.addTerm(-1);
    addMonitoring(network, monitoringRoot);
    run(log, monitoringRoot, monitor, trainingData, history, network);
  
    timeoutMinutes = 120;
    tree.addTerm(2);
    addMonitoring(network, monitoringRoot);
    run(log, monitoringRoot, monitor, trainingData, history, network);
  
    return network;
  }
  
  public void run(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history, PipelineNetwork network) {
    train(log, network, trainingData, monitor);
    report(log, monitoringRoot, history);
    validate(log, network);
    monitor.clear();
    history.clear();
  }
  
  @Override
  public PipelineNetwork buildModel(NotebookOutput log) {
    log.p("This is a very simple model that performs basic logistic regression. " +
            "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      this.tree = new PolynomialNetwork(
        new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}).setWeights(()->1e-8*(Math.random()-1))
      );
      network.add(tree);
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
}
