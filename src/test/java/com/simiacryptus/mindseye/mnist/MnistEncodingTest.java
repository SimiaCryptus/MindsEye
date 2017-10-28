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
import com.simiacryptus.mindseye.eval.GpuTrainable;
import com.simiacryptus.mindseye.eval.RepresentationTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestCategories;
import com.simiacryptus.util.text.TableOutput;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * The type Mnist test base.
 */
public class MnistEncodingTest {

  /**
   * Basic.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws IOException {
    PrintStream originalOut = System.out;
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) ((MarkdownNotebookOutput) log).addCopy(originalOut);
      List<Step> history = new ArrayList<>();
      TrainingMonitor monitor = getMonitor(originalOut, history);
      Tensor[][] trainingData = getTrainingData(log);
      NNLayer network = buildModel(log);
  
      Tensor[][] primingData = Arrays.copyOfRange(trainingData, 0, 1000);
      train(log, monitor, network, primingData);
      validate(log, network, primingData);
      report(log, history, network);
  
      train(log, monitor, network, trainingData);
      validate(log, network, trainingData);
      report(log, history, network);

    }
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
   * The Model no.
   */
  int modelNo = 0;

  /**
   * Report.
   *  @param log            the log
   * @param history        the history
   * @param network        the network
   */
  public void report(NotebookOutput log, List<Step> history, NNLayer network) {
    log.code(() -> {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      return out.toString();
    });
    String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
    if(!history.isEmpty()) log.code(() -> {
      PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Iteration", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    });
  }
  
  /**
   * Validate.
   *  @param log     the log
   * @param network the network
   * @param data
   */
  public void validate(NotebookOutput log, NNLayer network, Tensor[][] data) {
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).map(tensorArray -> {
        try {
          Tensor predictionSignal = CudaExecutionContext.gpuContexts.map(ctx->network.eval(ctx, tensorArray[0]).getData().get(0));
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Source", log.image(tensorArray[1].toGrayImage(), ""));
          row.put("Echo", log.image(predictionSignal.toGrayImage(), ""));
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  public Tensor[][] getTrainingData(NotebookOutput log) {
    return log.code(() -> {
      return getTrainingData(log, getTrainingStream());
    });
  }
  
  public Tensor[][] getValidationData(NotebookOutput log) {
    return log.code(() -> {
      return getTrainingData(log, getValidationStream());
    });
  }
  
  public Tensor[][] getTrainingData(NotebookOutput log, Stream<LabeledObject<Tensor>> labeledObjectStream) {
    return labeledObjectStream.map(labeledObject -> {
      return new Tensor[]{ new Tensor(100), labeledObject.data };
    }).toArray(i -> new Tensor[i][]);
  }
  
  public Stream<LabeledObject<Tensor>> getTrainingStream() {
    try {
      return MNIST.trainingDataStream();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  public Stream<LabeledObject<Tensor>> getValidationStream() {
    try {
      return MNIST.validationDataStream();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets training trainable.
   *
   * @param log               the log
   * @param data      the training data
   * @param network the supervised network
   * @return the training trainable
   */
  public Trainable getTrainingTrainable(NotebookOutput log, Tensor[][] data, SimpleLossNetwork network) {
    //Trainable trainable = new DeltaHoldoverArrayTrainable(data, supervisedNetwork, trainingSize);
    //printSample(log, expanded, 100);
    return new RepresentationTrainable(network, data, new boolean[]{ true, false });
  }
  
  /**
   * Build model pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public NNLayer buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork();
      //network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(100));
      network.add(new DenseSynapseLayer(new int[]{100}, new int[]{28, 28, 1})
                    .setWeights(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new BiasLayer(28, 28, 1));
      return network;
    });
  }
  
  public void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new MeanSqLossLayer());
      return new IterativeTrainer(getTrainingTrainable(log, data, supervisedNetwork))
               .setMonitor(monitor)
               .setOrientation(new QQN())
               .setLineSearchFactory(name->new QuadraticSearch()
                                             .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
                                             .setRelativeTolerance(2e-1))
               .setTimeout(60, TimeUnit.MINUTES)
               .setMaxIterations(10000)
               .run();
    });
  }
  
  
}
