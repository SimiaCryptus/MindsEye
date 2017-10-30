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
import com.simiacryptus.mindseye.eval.RepresentationTrainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.LogActivationLayer;
import com.simiacryptus.mindseye.layers.activation.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.meta.SignMetaLayer;
import com.simiacryptus.mindseye.layers.meta.SignMetaReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumReducerLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.network.graph.EvaluationContext;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
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

/**
 * The type Mnist test base.
 */
public class MnistEncodingTest {
  
  private int features = 10;
  
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
      DAGNetwork network = buildModel(log);
  
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
  public void validate(NotebookOutput log, DAGNetwork network, Tensor[][] data) {
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).map(tensorArray -> {
        try {
          Tensor predictionSignal = CudaExecutionContext.gpuContexts.run(ctx -> {
            EvaluationContext exeCtx = network.buildExeCtx(NNResult.singleResultArray(new Tensor[]{tensorArray[0], null}));
            return network.getByLabel("image").get(ctx, exeCtx).getData().get(0);
          });
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
      return MNIST.trainingDataStream().map(labeledObject -> {
        return new Tensor[]{ new Tensor(features).fill(()->0.5*(Math.random()-0.5)), labeledObject.data };
      }).toArray(i -> new Tensor[i][]);
    });
  }
  
  /**
   * Build model pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public DAGNetwork buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork(2);
      //network.add(new ReLuActivationLayer());
      //network.add(new BiasLayer(features));
      DAGNode input = network.getInput(0);
      network.add(new ReLuActivationLayer());
      network.add(new DenseSynapseLayer(new int[]{features}, new int[]{28, 28, 1})
                    .setWeights(() -> 0.25 * (Math.random() - 0.5)), input);
      network.add(new LinearActivationLayer());
      DAGNode image = network.add("image", new BiasLayer(28, 28, 1), network.getHead());
  
      network.add(new ProductInputsLayer(),
        network.add(new MeanSqLossLayer(), image, network.getInput(1)),
        network.add(new AvgReducerLayer(),
          network.add(new SignMetaReducerLayer(),
            input)));

      return network;
    });
  }
  
  public void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data) {
    log.code(() -> {
      //Trainable trainable = new DeltaHoldoverArrayTrainable(data, supervisedNetwork, trainingSize);
      //printSample(log, expanded, features);
      RepresentationTrainable subject = new RepresentationTrainable(network, data, new boolean[]{true, false});
      return new IterativeTrainer(subject)
               .setMonitor(monitor)
               .setOrientation(new QQN())
//               .setLineSearchFactory(name->new QuadraticSearch()
//                                             .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
//                                             .setRelativeTolerance(2e-1))
               .setTimeout(5, TimeUnit.MINUTES)
               .setMaxIterations(100)
               .run();
    });
  }
  
  
}
