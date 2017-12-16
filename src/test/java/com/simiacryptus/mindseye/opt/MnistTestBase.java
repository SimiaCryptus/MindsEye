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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.stream.IntStream;

/**
 * The type Mnist run base.
 */
public abstract class MnistTestBase {
  
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
  public void test() throws IOException {
    PrintStream originalOut = System.out;
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) log.addCopy(originalOut);
      List<Step> history = new ArrayList<>();
      MonitoredObject monitoringRoot = new MonitoredObject();
      TrainingMonitor monitor = getMonitor(originalOut, history);
      Tensor[][] trainingData = getTrainingData(log);
      _test(log, monitoringRoot, monitor, trainingData, history);
    }
  }
  
  /**
   * Test nn layer.
   *
   * @param log            the log
   * @param monitoringRoot the monitoring root
   * @param monitor        the monitor
   * @param trainingData   the training data
   * @param history        the history
   * @return the nn layer
   */
  public NNLayer _test(NotebookOutput log, MonitoredObject monitoringRoot, TrainingMonitor monitor, Tensor[][] trainingData, List<Step> history) {
    DAGNetwork network = buildModel(log);
    addMonitoring(network, monitoringRoot);
    log.h3("Training");
    train(log, network, trainingData, monitor);
    report(log, monitoringRoot, history, network);
    validate(log, network);
    return network;
  }
  
  /**
   * Gets monitor.
   *
   * @param originalOut the original out
   * @param history     the history
   * @return the monitor
   */
  public TrainingMonitor getMonitor(PrintStream originalOut, List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void log(String msg) {
        System.out.println(msg);
        if (null != originalOut && System.out != originalOut) originalOut.println(msg);
        super.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        history.add(currentPoint);
        super.onStepComplete(currentPoint);
      }
      
      @Override
      public void clear() {
        super.clear();
      }
    };
  }
  
  /**
   * Report.
   *
   * @param log            the log
   * @param monitoringRoot the monitoring root
   * @param history        the history
   * @param network        the network
   */
  public void report(NotebookOutput log, MonitoredObject monitoringRoot, List<Step> history, NNLayer network) {
    
    if (!history.isEmpty()) {
      log.code(() -> {
        PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }
    
    String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
    
    log.h3("Metrics");
    log.code(() -> {
      try {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonUtil.writeJson(out, monitoringRoot.getMetrics());
        return out.toString();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  /**
   * Add monitoring.
   *
   * @param network        the network
   * @param monitoringRoot the monitoring root
   */
  public void addMonitoring(DAGNetwork network, MonitoredObject monitoringRoot) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public void removeMonitoring(DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * Train.
   *
   * @param log          the log
   * @param network      the network
   * @param trainingData the training data
   * @param monitor      the monitor
   */
  public abstract void train(NotebookOutput log, NNLayer network, Tensor[][] trainingData, TrainingMonitor monitor);
  
  /**
   * Validate.
   *
   * @param log     the log
   * @param network the network
   */
  public void validate(NotebookOutput log, NNLayer network) {
    log.h3("Validation");
    log.p("If we run our model against the entire validation dataset, we get this accuracy:");
    log.code(() -> {
      return MNIST.validationDataStream().mapToDouble(labeledObject ->
        predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
        .average().getAsDouble() * 100;
    });
    
    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.code(() -> {
      try {
        TableOutput table = new TableOutput();
        MNIST.validationDataStream().map(labeledObject -> {
          try {
            int actualCategory = parse(labeledObject.label);
            double[] predictionSignal = GpuController.call(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
            int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
            if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
            LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
            row.put("Image", log.image(labeledObject.data.toGrayImage(), labeledObject.label));
            row.put("Prediction", Arrays.stream(predictionList).limit(3)
              .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
              .reduce((a, b) -> a + ", " + b).get());
            return row;
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }).filter(x -> null != x).limit(10).forEach(table::putRow);
        return table;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
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
    return IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
  }
  
  /**
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(NotebookOutput log) {
    try {
      return MNIST.trainingDataStream().map(labeledObject -> {
        Tensor categoryTensor = new Tensor(10);
        int category = parse(labeledObject.label);
        categoryTensor.set(category, 1);
        return new Tensor[]{labeledObject.data, categoryTensor};
      }).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Build model dag network.
   *
   * @param log the log
   * @return the dag network
   */
  public DAGNetwork buildModel(NotebookOutput log) {
    log.h3("Model");
    log.p("This is a very simple model that performs basic logistic regression. " +
      "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10})
        .setWeights(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
}
