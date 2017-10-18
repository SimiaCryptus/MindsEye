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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class SigmoidTreeNetworkTests {
  public static class LinearTest extends MnistTestBase {
    
    SigmoidTreeNetwork tree;
    int trainingSize = 10000;
    int iterationsPerSample = 25;
    int maxIterations = 500;
    int timeoutMinutes = 5;
    
    @Override
    public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
      log.code(() -> {
        SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
        StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, trainingSize);
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
      // Base Linear Phase 1
      run(log, monitoringRoot, monitor, trainingData, history, network);
      iterationsPerSample = 500;
      trainingSize = 0;
      timeoutMinutes = 120;
      // Base Linear Phase 2
      run(log, monitoringRoot, monitor, trainingData, history, network);
      timeoutMinutes = 120;
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 2; i++) {
          nextPhase(monitoringRoot, monitor, history, network);
          run(log, monitoringRoot, monitor, trainingData, history, network);
        }
        trainingSize *= 2;
        timeoutMinutes *= 2;
      }
      
      return network;
    }
    
    public void run(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history, PipelineNetwork network) {
      train(log, network, trainingData, monitor);
      report(log, monitoringRoot, history);
      validate(log, network);
      monitor.clear();
      history.clear();
    }
    
    public void nextPhase(MonitoredObject monitoringRoot, TrainingMonitor monitor, List<Step> history, PipelineNetwork network) {
      removeMonitoring(network);
      tree.nextPhase();
      addMonitoring(network, monitoringRoot);
    }
    
    @Override
    public PipelineNetwork buildModel(NotebookOutput log) {
      log.p("This is a very simple model that performs basic logistic regression. " +
              "It is expected to be trainable to about 91% accuracy on MNIST.");
      return log.code(() -> {
        PipelineNetwork network = new PipelineNetwork();
        this.tree = new SigmoidTreeNetwork(
                                            new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}),
                                            new BiasLayer(28, 28, 1)
        ).setSkipFuzzy(true).setSkipChildStage(true);
        network.add(tree);
        network.add(new SoftmaxActivationLayer());
        return network;
      });
    }
  }
  
  public static class ConvTest extends LinearTest {
    @Override
    public PipelineNetwork buildModel(NotebookOutput log) {
      log.p("This is a very simple model that performs basic logistic regression. " +
              "It is expected to be trainable to about 91% accuracy on MNIST.");
      return log.code(() -> {
        PipelineNetwork network = new PipelineNetwork();
        network.add(new ConvolutionLayer(3,3,5,false));
        network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Avg));
        this.tree = new SigmoidTreeNetwork(
                                            new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}),
                                            new BiasLayer(28, 28, 1)
        ).setSkipFuzzy(true).setSkipChildStage(true);
        network.add(tree);
        network.add(new SoftmaxActivationLayer());
        return network;
      });
    }
  }
  
}