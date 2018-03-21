/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Mnist apply base.
 */
public abstract class MnistTestBase extends NotebookReportBase {
  private static final Logger log = LoggerFactory.getLogger(MnistTestBase.class);
  
  /**
   * The Model no.
   */
  int modelNo = 0;
  
  /**
   * Test.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() {
    run(this::run);
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Optimizers;
  }
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    @Nonnull final List<Step> history = new ArrayList<>();
    @Nonnull final MonitoredObject monitoringRoot = new MonitoredObject();
    @Nonnull final TrainingMonitor monitor = getMonitor(history);
    final Tensor[][] trainingData = getTrainingData(log);
    final DAGNetwork network = buildModel(log);
    addMonitoring(network, monitoringRoot);
    log.h1("Training");
    train(log, network, trainingData, monitor);
    report(log, monitoringRoot, history, network);
    validate(log, network);
    removeMonitoring(network);
  }
  
  /**
   * Add monitoring.
   *
   * @param network        the network
   * @param monitoringRoot the monitoring root
   */
  public void addMonitoring(@Nonnull final DAGNetwork network, @Nonnull final MonitoredObject monitoringRoot) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }
  
  /**
   * Build model dag network.
   *
   * @param log the log
   * @return the dag network
   */
  public DAGNetwork buildModel(@Nonnull final NotebookOutput log) {
    log.h1("Model");
    log.p("This is a very simple model that performs basic logistic regression. " +
      "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.code(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10})
        .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(final NotebookOutput log) {
    Tensor[][] tensors = MNIST.trainingDataStream().map(labeledObject -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject.label);
      categoryTensor.set(category, 1);
      return new Tensor[]{labeledObject.data, categoryTensor};
    }).toArray(i -> new Tensor[i][]);
    return tensors;
  }
  
  /**
   * Parse int.
   *
   * @param label the label
   * @return the int
   */
  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }
  
  /**
   * Predict int [ ].
   *
   * @param network       the network
   * @param labeledObject the labeled object
   * @return the int [ ]
   */
  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    @Nullable final double[] predictionSignal = network.eval(labeledObject.data).getData().get(0).getData();
    return IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public void removeMonitoring(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * Report.
   *
   * @param log            the log
   * @param monitoringRoot the monitoring root
   * @param history        the history
   * @param network        the network
   */
  public void report(@Nonnull final NotebookOutput log, @Nonnull final MonitoredObject monitoringRoot, @Nonnull final List<Step> history, @Nonnull final Layer network) {
    
    if (!history.isEmpty()) {
      log.code(() -> {
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }
  
    @Nonnull final String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
  
    log.h1("Metrics");
    log.code(() -> {
      try {
        @Nonnull final ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonUtil.writeJson(out, monitoringRoot.getMetrics());
        return out.toString();
      } catch (@Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  @Nonnull
  public TrainingMonitor getMonitor(@Nonnull final List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }
      
      @Override
      public void log(final String msg) {
        log.info(msg);
        super.log(msg);
      }
      
      @Override
      public void onStepComplete(final Step currentPoint) {
        history.add(currentPoint);
        super.onStepComplete(currentPoint);
      }
    };
  }
  
  /**
   * Train.
   *
   * @param log          the log
   * @param network      the network
   * @param trainingData the training data
   * @param monitor      the monitor
   */
  public abstract void train(NotebookOutput log, Layer network, Tensor[][] trainingData, TrainingMonitor monitor);
  
  /**
   * Validate.
   *
   * @param log     the log
   * @param network the network
   */
  public void validate(@Nonnull final NotebookOutput log, @Nonnull final Layer network) {
    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.code(() -> {
      return MNIST.validationDataStream().mapToDouble(labeledObject ->
        predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
        .average().getAsDouble() * 100;
    });
    
    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.code(() -> {
      @Nonnull final TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        try {
          final int actualCategory = parse(labeledObject.label);
          @Nullable final double[] predictionSignal = network.eval(labeledObject.data).getData().get(0).getData();
          final int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
          @Nonnull final LinkedHashMap<String, Object> row = new LinkedHashMap<>();
          row.put("Image", log.image(labeledObject.data.toGrayImage(), labeledObject.label));
          row.put("Prediction", Arrays.stream(predictionList).limit(3)
            .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
            .reduce((a, b) -> a + ", " + b).get());
          return row;
        } catch (@Nonnull final IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
}
