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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The type Mnist encoding run.
 */
public class EncodingProblem extends ImageTestUtil implements Problem {
  
  private static int modelNo = 0;
  
  private final int batchSize = 10000;
  private final RevNetworkFactory revFactory;
  private final OptimizationStrategy optimizer;
  private final List<StepRecord> history = new ArrayList<>();
  private final ImageData data;
  private int timeoutMinutes = 1;
  private int features = 10;
  
  /**
   * Instantiates a new Encoding problem.
   *
   * @param revFactory the rev factory
   * @param optimizer  the optimizer
   * @param data       the data
   */
  public EncodingProblem(RevNetworkFactory revFactory, OptimizationStrategy optimizer, ImageData data) {
    this.revFactory = revFactory;
    this.optimizer = optimizer;
    this.data = data;
  }
  
  
  @Override
  public EncodingProblem run(NotebookOutput log) {
    TrainingMonitor monitor = getMonitor(history);
    Tensor[][] trainingData;
    try {
      trainingData = data.trainingData().map(labeledObject -> {
        return new Tensor[]{new Tensor(features).fill(() -> 0.5 * (Math.random() - 0.5)), labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    
    DAGNetwork imageNetwork = revFactory.vectorToImage(log, features);
    
    PipelineNetwork trainingNetwork = new PipelineNetwork(2);
    DAGNode image = trainingNetwork.add(imageNetwork, trainingNetwork.getInput(0));
    DAGNode softmax = trainingNetwork.add(new SoftmaxActivationLayer(), trainingNetwork.getInput(0));
    trainingNetwork.add(new SumInputsLayer(),
      trainingNetwork.add(new LinearActivationLayer().setScale(1).freeze(),
        trainingNetwork.add(new EntropyLossLayer(), softmax, softmax)),
      trainingNetwork.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
        trainingNetwork.add(new MeanSqLossLayer(), image, trainingNetwork.getInput(1))
      )
    );
    
    log.h3("Training");
    log.p("We start by training with a very small population to improve intial convergence performance:");
    Tensor[][] primingData = Arrays.copyOfRange(trainingData, 0, 1000);
    ValidatingTrainer preTrainer = optimizer.train(log,
      (SampledTrainable) new SampledArrayTrainable(primingData, trainingNetwork, 1000, batchSize).setMask(true, false),
      new ArrayTrainable(primingData, trainingNetwork, batchSize), monitor);
    log.code(() -> {
      preTrainer.setTimeout(timeoutMinutes / 2, TimeUnit.MINUTES).setMaxIterations(batchSize).run();
    });
    
    log.p("Then our main training phase:");
    ValidatingTrainer mainTrainer = optimizer.train(log,
      (SampledTrainable) new SampledArrayTrainable(trainingData, trainingNetwork, 1000, batchSize).setMask(true, false),
      new ArrayTrainable(trainingData, trainingNetwork, batchSize), monitor);
    log.code(() -> {
      mainTrainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES).setMaxIterations(batchSize).run();
    });
    
    if (!history.isEmpty()) {
      log.code(() -> {
        return plot(history);
      });
      log.code(() -> {
        return plotTime(history);
      });
    }
    String modelName = "encoding_model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(trainingNetwork.getJson().toString(), modelName, modelName));
    
    log.h3("Results");
    PipelineNetwork testNetwork = new PipelineNetwork(2);
    testNetwork.add(imageNetwork, testNetwork.getInput(0));
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(trainingData).map(tensorArray -> {
        try {
          Tensor predictionSignal = GpuController.call(ctx -> testNetwork.eval(ctx, tensorArray)).getData().get(0);
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Source", log.image(tensorArray[1].toImage(), ""));
          row.put("Echo", log.image(predictionSignal.toImage(), ""));
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
    
    log.p("Learned Model Statistics:");
    log.code(() -> {
      ScalarStatistics scalarStatistics = new ScalarStatistics();
      trainingNetwork.state().stream().flatMapToDouble(x -> Arrays.stream(x))
        .forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    
    log.p("Learned Representation Statistics:");
    log.code(() -> {
      ScalarStatistics scalarStatistics = new ScalarStatistics();
      Arrays.stream(trainingData)
        .flatMapToDouble(row -> Arrays.stream(row[0].getData()))
        .forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    return this;
  }
  
  
  /**
   * Gets timeout minutes.
   *
   * @return the timeout minutes
   */
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }
  
  /**
   * Sets timeout minutes.
   *
   * @param timeoutMinutes the timeout minutes
   * @return the timeout minutes
   */
  public EncodingProblem setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  /**
   * Gets features.
   *
   * @return the features
   */
  public int getFeatures() {
    return features;
  }
  
  /**
   * Sets features.
   *
   * @param features the features
   * @return the features
   */
  public EncodingProblem setFeatures(int features) {
    this.features = features;
    return this;
  }
  
  @Override
  public List<StepRecord> getHistory() {
    return history;
  }
}
