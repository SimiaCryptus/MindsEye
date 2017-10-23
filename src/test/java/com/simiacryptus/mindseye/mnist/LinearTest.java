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

import com.simiacryptus.mindseye.data.MNIST;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.trainable.StaticArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

public class LinearTest extends MnistTestBase {
  
  protected String description = "This report demonstrates a basic linear model fit against the MNIST dataset. " +
                         "It serves as a reference report to compare algorithm variants. " +
                         "This is a very simple model that performs basic logistic regression. " +
                         "It is expected to be trainable to about 91% accuracy on MNIST.";

  @Override
  public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      return new ValidatingTrainer(
        getTrainingTrainable(log, trainingData, supervisedNetwork),
        getValidationTrainable(supervisedNetwork))
               .setMaxEpochIterations(20)
               .setTrainingTarget(0.7)
               .setMaxTrainingSize(100000)
               .setMonitor(monitor)
               .setOrientation(new QQN())
               .setLineSearchFactory(name->new QuadraticSearch()
                                             .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
                                             .setRelativeTolerance(2e-1))
               .setTimeout(8, TimeUnit.HOURS)
               .setMaxIterations(1000)
               .run();
    });
  }
  
  public StochasticArrayTrainable getTrainingTrainable(NotebookOutput log, Tensor[][] trainingData, SimpleLossNetwork supervisedNetwork) {
      //Trainable trainable = new DeltaHoldoverArrayTrainable(trainingData, supervisedNetwork, trainingSize);
    Tensor[][] expanded = Arrays.stream(trainingData).flatMap(row -> expand(row)).toArray(i -> new Tensor[i][]);
    printSample(log, expanded, 100);
    return new StochasticArrayTrainable(expanded, supervisedNetwork, 10000, 50000);
  }
  
  public Trainable getValidationTrainable(SimpleLossNetwork supervisedNetwork) {
    Tensor[][] validationData = new Tensor[0][];
    try {
      validationData = MNIST.validationDataStream().map(labeledObject -> {
        Tensor categoryTensor = new Tensor(10);
        int category = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
      }).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return new StaticArrayTrainable(validationData, supervisedNetwork, 50000);
    //return new StochasticArrayTrainable(validationData, supervisedNetwork, 50000);
  }
  
  public static void printSample(NotebookOutput log, Tensor[][] expanded, int size) {
    ArrayList<Tensor[]> list = new ArrayList<>(Arrays.asList(expanded));
    Collections.shuffle(list);
    log.p("Expanded Training Data Sample: " + list.stream().limit(size).map(x -> {
      try {
        return log.image(x[0].toGrayImage(), "");
      } catch (IOException e) {
        e.printStackTrace();
        return "";
      }
    }).reduce((a,b)->a+b).get());
  }

  protected Stream<Tensor[]> expand(Tensor... data) {
    Random random = new Random();
    return Stream.of(
      new Tensor[]{ data[0], data[1] },
      new Tensor[]{ addNoise(data[0]), data[1] },
      new Tensor[]{ translate(random.nextInt(5)-3, random.nextInt(5)-3, data[0]), data[1] },
      new Tensor[]{ translate(random.nextInt(5)-3, random.nextInt(5)-3, data[0]), data[1] }
    );
  }

  protected static Tensor translate(int dx, int dy, Tensor tensor) {
    int sx = tensor.getDimensions()[0];
    int sy = tensor.getDimensions()[1];
    return new Tensor(tensor.getDimensions(), tensor.coordStream(false).mapToDouble(c -> {
      int x = c.coords[0] + dx;
      int y = c.coords[1] + dy;
      if(x < 0 || x >= sx) {
        return 0.0;
      } else if(y < 0 || y >= sy) {
        return 0.0;
      } else {
        return tensor.get(x, y);
      }
    }).toArray());
  }

  protected static Tensor addNoise(Tensor tensor) {
    return tensor.mapParallel((v)-> Math.random()<0.9?v:(v + Math.random() * 100));
  }
  
  @Override
  public PipelineNetwork buildModel(NotebookOutput log) {
    log.p("");
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10}).setWeights(()->1e-8*(Math.random()-0.5)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
  public void run(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history, PipelineNetwork network) {
    train(log, network, trainingData, monitor);
    report(log, monitoringRoot, history, network);
    validate(log, network);
    monitor.clear();
    history.clear();
  }
}
