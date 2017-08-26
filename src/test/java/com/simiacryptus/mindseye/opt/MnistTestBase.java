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

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.InnerNode;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.test.MNIST;
import com.simiacryptus.util.test.TestCategories;
import com.simiacryptus.util.text.TableOutput;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

public abstract class MnistTestBase {
  /**
   * Basic.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this).addCopy(System.out)) {
      log.p("First, define a model:");
      PipelineNetwork network = buildModel(log);
      Tensor[][] trainingData = getTrainingData(log);
      MonitoredObject monitoringRoot = new MonitoredObject();
      List<Step> history = new ArrayList<>();
      addMonitoring(network, monitoringRoot);
      TrainingMonitor monitor = new TrainingMonitor() {
        @Override
        public void log(String msg) {
          System.out.println(msg);
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
      train(log, network, trainingData, monitor);
      validate(log, network);
      report(log, monitoringRoot, history);
    }
  }
  
  public void report(NotebookOutput log, MonitoredObject monitoringRoot, List<Step> history) {
    log.code(()->{
      try {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonUtil.writeJson(out, monitoringRoot.getMetrics());
        return out.toString();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    log.code(() ->{
      PlotCanvas plot = ScatterPlot.plot(history.stream().map(step->new double[]{step.iteration, Math.log10(step.point.value)}).toArray(i->new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Iteration", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    });
  }
  
  public void addMonitoring(PipelineNetwork network, MonitoredObject monitoringRoot) {
    network.visitNodes(node->{
      if(node instanceof InnerNode && !(node.getLayer() instanceof MonitoringWrapper)) {
        ((InnerNode)node).setLayer(new MonitoringWrapper(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }
  
  public abstract void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor);
  
  /**
   * Validate.
   *
   * @param log     the log
   * @param network the network
   */
  public void validate(NotebookOutput log, PipelineNetwork network) {
    log.p("If we test our model against the entire validation dataset, we get this accuracy:");
    log.code(() -> {
      try {
        return MNIST.validationDataStream().mapToDouble(labeledObject -> {
          int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
          double[] predictionSignal = network.eval(new NNLayer.NNExecutionContext() {
          }, labeledObject.data).getData().get(0).getData();
          int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          return predictionList[0] == actualCategory ? 1 : 0;
        }).average().getAsDouble() * 100;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    
    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.code(() -> {
      try {
        TableOutput table = new TableOutput();
        MNIST.validationDataStream().map(labeledObject -> {
          try {
            int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
            double[] predictionSignal = network.eval(new NNLayer.NNExecutionContext() {
            }, labeledObject.data).getData().get(0).getData();
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
   * Get training data tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] getTrainingData(NotebookOutput log) {
    log.p("We use the standard MNIST dataset, made available by a helper function. " +
            "In order to use data, we convert it into data tensors; helper functions are defined to " +
            "work map images.");
    return log.code(() -> {
      try {
        return MNIST.trainingDataStream().map(labeledObject -> {
          Tensor categoryTensor = new Tensor(10);
          int category = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
          categoryTensor.set(category, 1);
          return new Tensor[]{labeledObject.data, categoryTensor};
        }).toArray(i -> new Tensor[i][]);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  /**
   * Build model pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork buildModel(NotebookOutput log) {
    log.p("This is a very simple model that performs basic logistic regression. " +
            "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new DenseSynapseLayer(new int[]{28, 28, 1}, new int[]{10})
                    .setWeights(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
}
