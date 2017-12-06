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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * The type Mnist run base.
 */
public class ClassifyProblem extends ImageTestUtil implements Problem {
  
  private static int modelNo = 0;
  
  private final int batchSize = 10000;
  private final int categories;
  private final FwdNetworkFactory fwdFactory;
  private final OptimizationStrategy optimizer;
  private final List<StepRecord> history = new ArrayList<>();
  private final ImageData data;
  private int timeoutMinutes = 1;
  
  /**
   * Instantiates a new Classify problem.
   *
   * @param fwdFactory the fwd factory
   * @param optimizer  the optimizer
   * @param data       the data
   * @param categories the categories
   */
  public ClassifyProblem(FwdNetworkFactory fwdFactory, OptimizationStrategy optimizer, ImageData data, int categories) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.data = data;
    this.categories = categories;
  }
  
  
  public ClassifyProblem run(NotebookOutput log) {
    MonitoredObject monitoringRoot = new MonitoredObject();
    TrainingMonitor monitor = getMonitor(history);
    Tensor[][] trainingData = getTrainingData(log);
    DAGNetwork network = fwdFactory.imageToVector(log, categories);
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    addMonitoring(network, monitoringRoot);
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
    String modelName = "classification_model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
    log.h3("Metrics");
    log.code(() -> {
      return toFormattedJson(monitoringRoot.getMetrics());
    });
    
    log.h3("Validation");
    log.p("If we run our model against the entire validation dataset, we get this accuracy:");
    log.code(() -> {
      return data.validationData().mapToDouble(labeledObject ->
        predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
        .average().getAsDouble() * 100;
    });
    
    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.code(() -> {
      try {
        TableOutput table = new TableOutput();
        data.validationData().map(labeledObject -> {
          return toRow(log, labeledObject, GpuController.call(ctx -> network.eval(ctx, labeledObject.data)).getData().get(0).getData());
        }).filter(x -> null != x).limit(10).forEach(table::putRow);
        return table;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
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
      int actualCategory = parse(labeledObject.label);
      int[] predictionList = IntStream.range(0, categories).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
      if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
      LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
      row.put("Image", log.image(labeledObject.data.toImage(), labeledObject.label));
      row.put("Prediction", Arrays.stream(predictionList).limit(3)
        .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
        .reduce((a, b) -> a + ", " + b).get());
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
   * Predict int [ ].
   *
   * @param network       the network
   * @param labeledObject the labeled object
   * @return the int [ ]
   */
  public int[] predict(NNLayer network, LabeledObject<Tensor> labeledObject) {
    double[] predictionSignal = GpuController.call(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
    return IntStream.range(0, categories).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
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
        Tensor categoryTensor = new Tensor(categories);
        int category = parse(labeledObject.label);
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
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
  public ClassifyProblem setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  public List<StepRecord> getHistory() {
    return history;
  }
}
