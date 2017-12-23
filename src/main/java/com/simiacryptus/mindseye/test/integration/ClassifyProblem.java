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

package com.simiacryptus.mindseye.test.integration;

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Mnist run base.
 */
public class ClassifyProblem implements Problem {
  
  private static int modelNo = 0;
  
  private int batchSize = 10000;
  private final int categories;
  private final ImageProblemData data;
  private final FwdNetworkFactory fwdFactory;
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final List<String> labels;
  private int timeoutMinutes = 1;
  
  /**
   * Instantiates a new Classify problem.
   *
   * @param fwdFactory the fwd factory
   * @param optimizer  the optimizer
   * @param data       the data
   * @param categories the categories
   */
  public ClassifyProblem(final FwdNetworkFactory fwdFactory, final OptimizationStrategy optimizer, final ImageProblemData data, final int categories) {
    this.fwdFactory = fwdFactory;
    this.optimizer = optimizer;
    this.data = data;
    this.categories = categories;
    try {
      this.labels = Stream.concat(this.data.trainingData(), this.data.validationData()).map(x -> x.label).distinct().sorted().collect(Collectors.toList());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  
  @Override
  public List<StepRecord> getHistory() {
    return history;
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
  public ClassifyProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(final NotebookOutput log) {
    try {
      return data.trainingData().map(labeledObject -> {
        final Tensor categoryTensor = new Tensor(categories);
        final int category = parse(labeledObject.label);
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
      }).toArray(i -> new Tensor[i][]);
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Parse int.
   *
   * @param label the label
   * @return the int
   */
  public int parse(final String label) {
    return this.labels.indexOf(label);
  }
  
  /**
   * Predict int [ ].
   *
   * @param network       the network
   * @param labeledObject the labeled object
   * @return the int [ ]
   */
  public int[] predict(final NNLayer network, final LabeledObject<Tensor> labeledObject) {
    final double[] predictionSignal = GpuController.call(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
    return IntStream.range(0, categories).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
  }
  
  @Override
  public ClassifyProblem run(final NotebookOutput log) {
    final TrainingMonitor monitor = TestUtil.getMonitor(history);
    final Tensor[][] trainingData = getTrainingData(log);
    
    final DAGNetwork network = fwdFactory.imageToVector(log, categories);
    log.h3("Network Diagram");
    log.code(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(network))
        .height(400).width(600).render(Format.PNG).toImage();
    });

    log.h3("Training");
    final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    TestUtil.instrumentPerformance(log, supervisedNetwork);
    int initialSampleSize = Math.max(trainingData.length / 5, Math.min(10, trainingData.length / 2));
    final ValidatingTrainer trainer = optimizer.train(log,
      new SampledArrayTrainable(trainingData, supervisedNetwork, initialSampleSize, getBatchSize()),
      new ArrayTrainable(trainingData, supervisedNetwork, getBatchSize()), monitor);
    log.code(() -> {
      trainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES).setMaxIterations(10000).run();
    });
    if (!history.isEmpty()) {
      log.code(() -> {
        return TestUtil.plot(history);
      });
      log.code(() -> {
        return TestUtil.plotTime(history);
      });
    }
    TestUtil.extractPerformance(log, supervisedNetwork);
    final String modelName = "classification_model" + ClassifyProblem.modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));

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
        final TableOutput table = new TableOutput();
        Lists.partition(data.validationData().collect(Collectors.toList()), 100).stream().flatMap(batch -> {
          TensorArray batchIn = new TensorArray(batch.stream().map(x -> x.data).toArray(i -> new Tensor[i]));
          TensorList batchOut = GpuController.call(ctx -> network.eval(ctx, new NNConstant(batchIn))).getData();
          return IntStream.range(0, batchOut.length())
            .mapToObj(i -> toRow(log, batch.get(i), batchOut.get(i).getData()));
        }).filter(x -> null != x).limit(10).forEach(table::putRow);
        return table;
      } catch (final IOException e) {
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
  public LinkedHashMap<String, Object> toRow(final NotebookOutput log, final LabeledObject<Tensor> labeledObject, final double[] predictionSignal) {
    try {
      final int actualCategory = parse(labeledObject.label);
      final int[] predictionList = IntStream.range(0, categories).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
      if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
      final LinkedHashMap<String, Object> row = new LinkedHashMap<>();
      row.put("Image", log.image(labeledObject.data.toImage(), labeledObject.label));
      row.put("Prediction", Arrays.stream(predictionList).limit(3)
        .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
        .reduce((a, b) -> a + ", " + b).get());
      return row;
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets batch size.
   *
   * @return the batch size
   */
  public int getBatchSize() {
    return batchSize;
  }
  
  /**
   * Sets batch size.
   *
   * @param batchSize the batch size
   * @return the batch size
   */
  public ClassifyProblem setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }
}
