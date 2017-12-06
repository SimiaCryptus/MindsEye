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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.DropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The type Mnist run base.
 */
public class AutoencodingProblem extends ImageTestUtil implements Problem {
  
  private static int modelNo = 0;
  
  private final int batchSize = 10000;
  private final FwdNetworkFactory fwdFactory;
  private final OptimizationStrategy optimizer;
  private final RevNetworkFactory revFactory;
  private final List<StepRecord> history = new ArrayList<>();
  private final ImageData data;
  private int timeoutMinutes = 1;
  
  /**
   * Instantiates a new Autoencoding problem.
   *
   * @param fwdFactory the fwd factory
   * @param optimizer  the optimizer
   * @param revFactory the rev factory
   * @param data       the data
   */
  public AutoencodingProblem(FwdNetworkFactory fwdFactory, OptimizationStrategy optimizer, RevNetworkFactory revFactory, ImageData data) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.revFactory = revFactory;
    this.data = data;
  }
  
  public AutoencodingProblem run(NotebookOutput log) {
    MonitoredObject monitoringRoot = new MonitoredObject();
    
    int features = 100;
    DAGNetwork fwdNetwork = fwdFactory.imageToVector(log, features);
    DAGNetwork revNetwork = revFactory.vectorToImage(log, features);
    
    PipelineNetwork echoNetwork = new PipelineNetwork(1);
    echoNetwork.add(fwdNetwork);
    echoNetwork.add(revNetwork);
    
    PipelineNetwork supervisedNetwork = new PipelineNetwork(1);
    supervisedNetwork.add(fwdNetwork);
    DropoutNoiseLayer dropoutNoiseLayer = new DropoutNoiseLayer().setValue(0.8);
    supervisedNetwork.add(dropoutNoiseLayer);
    supervisedNetwork.add(revNetwork);
    supervisedNetwork.add(new MeanSqLossLayer(),
      supervisedNetwork.getHead(),
      supervisedNetwork.getInput(0));
    
    TrainingMonitor monitor = new TrainingMonitor() {
      TrainingMonitor inner = getMonitor(history);
      
      @Override
      public void log(String msg) {
        inner.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        dropoutNoiseLayer.shuffle();
        inner.onStepComplete(currentPoint);
      }
    };
    
    Tensor[][] trainingData = getTrainingData(log);
    addMonitoring(supervisedNetwork, monitoringRoot);
    
    log.h3("Training");
    ValidatingTrainer trainer = optimizer.train(log,
      new SampledArrayTrainable(trainingData, supervisedNetwork, trainingData.length / 2, batchSize),
      new ArrayTrainable(trainingData, supervisedNetwork, batchSize), monitor);
    log.code(() -> {
      trainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES).setMaxIterations(10000).run();
    });
    if (!history.isEmpty()) {
      log.code(() -> {
        return plot(history);
      });
      log.code(() -> {
        return plotTime(history);
      });
    }
    
    {
      String modelName = "encoder_model" + modelNo++ + ".json";
      log.p("Saved model as " + log.file(fwdNetwork.getJson().toString(), modelName, modelName));
    }
    
    {
      String modelName = "decoder_model" + modelNo++ + ".json";
      log.p("Saved model as " + log.file(revNetwork.getJson().toString(), modelName, modelName));
    }
    
    log.h3("Metrics");
    log.code(() -> {
      return toFormattedJson(monitoringRoot.getMetrics());
    });
    
    log.h3("Validation");
    
    log.p("Here are some re-encoded examples:");
    log.code(() -> {
      TableOutput table = new TableOutput();
      data.validationData().map(labeledObject -> {
        return toRow(log, labeledObject, GpuController.call(ctx -> echoNetwork.eval(ctx, labeledObject.data)).getData().get(0).getData());
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
    
    log.p("Some rendered unit vectors:");
    for (int featureNumber = 0; featureNumber < features; featureNumber++) {
      try {
        Tensor input = new Tensor(features).set(featureNumber, 1);
        Tensor tensor = GpuController.call(ctx -> revNetwork.eval(ctx, input)).getData().get(0);
        log.out(log.image(tensor.toImage(), ""));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return this;
  }
  
  /**
   * To row linked hash map.
   *
   * @param log              the log
   * @param labeledObject    the labeled object
   * @param predictionSignal the prediction signal
   * @return the linked hash map
   */
  public LinkedHashMap<String, Object> toRow(NotebookOutput log, LabeledObject<Tensor> labeledObject, double[] predictionSignal) {
    try {
      LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
      row.put("Image", log.image(labeledObject.data.toImage(), labeledObject.label));
      row.put("Echo", log.image(new Tensor(predictionSignal, labeledObject.data.getDimensions()).toImage(), labeledObject.label));
      return row;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Parse int.
   *
   * @param label the label
   * @return the int
   */
  public int parse(String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(NotebookOutput log) {
    try {
      return data.trainingData().map(labeledObject -> {
        return new Tensor[]{labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
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
  public AutoencodingProblem setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  public List<StepRecord> getHistory() {
    return history;
  }
}
