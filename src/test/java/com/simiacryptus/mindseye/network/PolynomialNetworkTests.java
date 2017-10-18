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
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.meta.AvgNormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.opt.*;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.trainable.ArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class PolynomialNetworkTests {
  public static class LinearTest extends MnistTestBase {
    
    PolynomialNetwork tree;
    int trainingSize = 10000;
    int iterationsPerSample = 25;
    int maxIterations = 500;
    int timeoutMinutes = 5;
    
    @Override
    public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
      log.code(() -> {
        SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
        //Trainable trainable = new DeltaHoldoverArrayTrainable(trainingData, supervisedNetwork, trainingSize);
        StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, trainingSize);
        Trainable validation = new ArrayTrainable(trainingData, supervisedNetwork);
        return new ValidatingTrainer(trainable, validation)
                 .setEpochIterations(iterationsPerSample)
                 .setMonitor(monitor)
                 .setOrientation(new QQN())
                 .setLineSearchFactory(name->new QuadraticSearch()
                                               .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
                                               .setRelativeTolerance(2e-1))
                 .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
                 .setMaxIterations(maxIterations)
                 .run();
      });
      
    }
    
    @Override
    public PipelineNetwork _test(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history) {
      log.p("First, define a model:");
      PipelineNetwork network = buildModel(log);
  
      iterationsPerSample = 1;
      trainingSize = 2500;
      timeoutMinutes = 60;
      run(log, monitoringRoot, monitor, trainingData, history, network);
  
//      iterationsPerSample = 10;
//      trainingSize = 5000;
//      timeoutMinutes = 240;
//      run(log, monitoringRoot, monitor, trainingData, history, network);
//
////    iterationsPerSample = 0;
////    trainingSize = 0;
////      trainingSize = 10000;
//      timeoutMinutes = 120;
//      run(log, monitoringRoot, monitor, trainingData, history, network);
//      timeoutMinutes = 120;
//      run(log, monitoringRoot, monitor, trainingData, history, network);
      
      //timeoutMinutes = 60;
//      trainingSize = 5000;
      tree.addTerm(1);
      run(log, monitoringRoot, monitor, trainingData, history, network);
      //trainingSize = 10000;
      //run(log, monitoringRoot, monitor, trainingData, history, network);
      
      //timeoutMinutes = 120;
      tree.addTerm(-1);
      run(log, monitoringRoot, monitor, trainingData, history, network);
      
      //timeoutMinutes = 120;
      tree.addTerm(2);
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
        this.tree = new PolynomialNetwork(new int[]{28, 28, 1}, new int[]{10});
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
        this.tree = new PolynomialConvolutionNetwork(new int[]{28, 28, 1}, new int[]{26, 26, 5}, 3, false);
        network.add(this.tree);
        network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Avg));
        network.add(new AvgNormalizationMetaLayer());
        network.add(new DenseSynapseLayer(new int[]{13, 13, 5}, new int[]{10}).setWeights(()->1e-8*(Math.random()-0.5)));
        network.add(new SoftmaxActivationLayer());
        return network;
      });
    }
    
  }
  
}
