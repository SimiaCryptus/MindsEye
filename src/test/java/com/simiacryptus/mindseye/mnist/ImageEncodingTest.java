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

import com.simiacryptus.mindseye.data.Caltech101;
import com.simiacryptus.mindseye.eval.*;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.activation.*;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.media.ImgReshapeLayer;
import com.simiacryptus.mindseye.layers.meta.TargetValueLayer;
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.OwlQn;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Mnist test base.
 */
public class ImageEncodingTest {
  
  int fromSize = 0;
  int toSize = 0;

  /**
   * Basic test.
   *
   * @throws Exception any exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    PrintStream originalOut = System.out;
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) ((MarkdownNotebookOutput) log).addCopy(originalOut);
      List<Step> history = new ArrayList<>();
      TrainingMonitor monitor = getMonitor(originalOut, history);
  
      int timeoutMinutes = 30;
      int band1 = 3;
      toSize = fromSize = 256;
      Tensor[][] trainingData = getImages(log, new String[]{"kangaroo"}, fromSize);
  
      int band2 = 12;
      DAGNetwork innerModelA = log.code(() -> {
        int radius = 5;
        fromSize = toSize;
        toSize = (fromSize / 2 + (radius-1)); // 132
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(new ConvolutionLayer(radius, radius, band2, band1 * 4, false).setWeights(()-> 0.1 * (Math.random() - 0.5)));
        network.add(new ImgBandBiasLayer(band1 * 4));
        network.add(new ImgReshapeLayer(2,2,true));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        return network;
      });
      trainingData = Arrays.stream(trainingData).limit(50).map(x -> {
        return new Tensor[]{
          x[0],
          new Tensor(toSize, toSize, band2).fill(() -> 0.5 * (Math.random() - 0.5))
        };
      }).toArray(i -> new Tensor[i][]);
      {
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModelA, 0, 1, 0, 0);
        DAGNetwork trainingModel1 = buildTrainingModel(log, innerModelA, 0, 1, 1e1, 1e0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), timeoutMinutes, 0, false, true);
        validationReport(log, trainingData, innerModelA);
        printModel(log, innerModelA);
        printDataStatistics(log, trainingData);
        printHistory(log, history);
        history.clear();
        train(log, monitor, trainingModel1, trainingData, new OwlQn(), timeoutMinutes, 1e-3, false, true);
        validationReport(log, trainingData, innerModelA);
        printModel(log, innerModelA);
        printDataStatistics(log, trainingData);
        printHistory(log, history);
        history.clear();
      }
  
      int band3 = 48;
      DAGNetwork innerModelB = log.code(() -> {
        int radius = 5;
        fromSize = toSize;
        toSize = (fromSize / 2 + (radius-1)); // 70
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(new ConvolutionLayer(radius, radius, band3, band2 * 4, false).setWeights(()-> 0.1 * (Math.random() - 0.5)));
        network.add(new ImgBandBiasLayer(band2 * 4));
        network.add(new ImgReshapeLayer(2,2,true));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        return network;
      });
      DAGNetwork innerModelAB = log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(innerModelB);
        network.add(innerModelA);
        return network;
      });
      trainingData = Arrays.stream(trainingData).map(x -> {
        return new Tensor[]{
          x[0], x[1],
          new Tensor(toSize, toSize, band3).fill(() -> 0.5 * (Math.random() - 0.5))
        };
      }).toArray(i -> new Tensor[i][]);
      {
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModelB, 1, 2, 0, 0);
        DAGNetwork trainingModel1 = buildTrainingModel(log, innerModelAB, 0, 2, 1e-1, 0);
        DAGNetwork trainingModel2 = buildTrainingModel(log, innerModelAB, 0, 2, 1e0, 1e-1);
        train(log, monitor, trainingModel0, trainingData, new QQN(), timeoutMinutes, 0, false, false, true);
        validationReport(log, trainingData, innerModelA, innerModelB);
        printModel(log, innerModelB);
        printDataStatistics(log, trainingData);
        printHistory(log, history);
        history.clear();
        train(log, monitor, trainingModel1, trainingData, new OwlQn(), timeoutMinutes, 1e-4, false, false, true);
        validationReport(log, trainingData, innerModelA, innerModelB);
        printModel(log, innerModelB);
        printDataStatistics(log, trainingData);
        printHistory(log, history);
        history.clear();
        train(log, monitor, trainingModel2, trainingData, new OwlQn(), timeoutMinutes, 1e-5, false, false, true);
        validationReport(log, trainingData, innerModelA, innerModelB);
        printModel(log, innerModelB);
        printDataStatistics(log, trainingData);
        printHistory(log, history);
        history.clear();
      }
  
    }
  }
  
  private DAGNetwork buildTrainingModel(NotebookOutput log, DAGNetwork innerModel, int reproducedColumn, int learnedColumn, double factor_l1, double factor_entropy) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork(Math.max(learnedColumn, reproducedColumn)+1);
      DAGNode input = network.getInput(learnedColumn);
      DAGNode output = network.add("image", innerModel, input);
      DAGNode rmsError = network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
        network.add(new MeanSqLossLayer(), output, network.getInput(reproducedColumn))
      );
      List<DAGNode> fitnessNodes = new ArrayList<>();
      fitnessNodes.add(rmsError);
      if(0<factor_entropy) {
        DAGNode density = network.add(new L1NormalizationLayer(),
          network.add(new SigmoidActivationLayer().setBalanced(true),
            network.add(new AbsActivationLayer(), input)));
        DAGNode entropy = network.add(new AbsActivationLayer(),
          network.add(new EntropyLossLayer(), density, density));
        fitnessNodes.add(network.add(new LinearActivationLayer().setScale(factor_entropy).freeze(), entropy));
      }
      if(0<factor_l1) {
        double lfactor = 1.0;
        DAGNode avgSignal = network.add(new NthPowerActivationLayer().setPower(1.0/lfactor),
          network.add(new AvgReducerLayer(),
            network.add(new NthPowerActivationLayer().setPower(lfactor),
              input)));
        fitnessNodes.add(network.add(new LinearActivationLayer().setScale(factor_l1).freeze(),
          network.add(new MeanSqLossLayer(),
            network.add(new NthPowerActivationLayer().setPower(1), avgSignal),
            network.add(new NthPowerActivationLayer().setPower(0), avgSignal)
          )));
      }
      network.add(new SumInputsLayer(), fitnessNodes.toArray(new DAGNode[]{}));
      return network;
    });
  }
  
  private TrainingMonitor getMonitor(PrintStream originalOut, List<Step> history) {
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
  
  private void printDataStatistics(NotebookOutput log, Tensor[][] data) {
    for(int col=0;col<data[0].length;col++) {
      int c = col;
      log.out("Learned Representation Statistics for Column " + col);
      log.code(()->{
        ScalarStatistics scalarStatistics = new ScalarStatistics();
        Arrays.stream(data)
          .flatMapToDouble(row-> Arrays.stream(row[c].getData()))
          .forEach(v->scalarStatistics.add(v));
        return scalarStatistics.getMetrics();
      });
    }
  }
  
  int modelNo = 0;
  private void printModel(NotebookOutput log, NNLayer network) {
    log.out("Learned Model Statistics:");
    log.code(()->{
      ScalarStatistics scalarStatistics = new ScalarStatistics();
      network.state().stream().flatMapToDouble(x-> Arrays.stream(x))
        .forEach(v->scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
  }
  
  private void printHistory(NotebookOutput log, List<Step> history) {
    if(!history.isEmpty()) log.code(() -> {
      PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Iteration", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    });
  }
  
  private void validationReport(NotebookOutput log, Tensor[][] data, final NNLayer... network) {
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).map(tensorArray -> {
        try {
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          for(int col=0;col<tensorArray.length;col++) {
            Tensor tensor = tensorArray[col];
            row.put("Data_"+col, render(log, tensor));
            if(network.length >= col && 0 < col) {
              PipelineNetwork decoder = new PipelineNetwork();
              for(int i=col-1;i>=0;i--) {
                decoder.add(network[i]);
              }
              row.put("Decode_"+col, log.image(CudaExecutionContext.gpuContexts.run(ctx -> {
                return decoder.eval(ctx, new Tensor[]{tensor});
              }).getData().get(0).toImage(), ""));
            }
          }
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  public String render(NotebookOutput log, Tensor tensor) {
    DoubleStatistics statistics = new DoubleStatistics();
    statistics.accept(tensor.getData());
    return tensor.map(x -> 0xFF * (x - statistics.getMin()) / (statistics.getMax() - statistics.getMin())).toImages().stream().map(image -> {
      try {
        return log.image(image, "");
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).reduce((a, b) -> a + b).get();
  }
  
  private BufferedImage resize(BufferedImage source, int size) {
    BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC));
    graphics.drawImage(source, 0, 0, size, size, null);
    return image;
  }
  
  private Tensor[][] getImages(NotebookOutput log, String[] categories, int size) {
    log.code(() -> {
      return Caltech101.trainingDataStream().collect(Collectors.groupingBy(x -> x.label, Collectors.counting()));
    });
    int seed = (int)((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return log.code(() -> {
      return Caltech101.trainingDataStream().filter(x -> {
        return Arrays.asList(categories).contains(x.label);
      }).map(labeledObj -> new Tensor[]{
        Tensor.fromRGB(resize(labeledObj.data.get(), size))
      }).sorted(Comparator.comparingInt(a -> System.identityHashCode(a) ^ seed)).toArray(i -> new Tensor[i][]);
    });
  }
  
  private void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, double factor_l1, boolean... mask) {
    log.code(() -> {
      StochasticTrainable trainingSubject = new StochasticArrayTrainable(data, network, data.length);
      if(0 < factor_l1) trainingSubject = new ConstL12Normalizer(trainingSubject).setFactor_L1(factor_l1);
      trainingSubject = (StochasticTrainable) ((TrainableDataMask) trainingSubject).setMask(mask);
      new ValidatingTrainer(trainingSubject, new ArrayTrainable(data, network))
        .setMaxTrainingSize(data.length)
        .setMinTrainingSize(1)
        .setMonitor(monitor)
        .setOrientation(orientation)
        .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
        .setMaxIterations(1000)
        .run();
    });
  }
  
  
}
