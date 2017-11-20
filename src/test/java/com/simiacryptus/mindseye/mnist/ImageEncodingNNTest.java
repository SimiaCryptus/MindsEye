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

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.mindseye.eval.StochasticTrainable;
import com.simiacryptus.mindseye.eval.TrainableDataMask;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.orient.QuantifyOrientationWrapper;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.io.HtmlNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * The type Image encoding nn run.
 */
public class ImageEncodingNNTest extends ImageEncodingPCATest {
  
  /**
   * The Pretrain minutes.
   */
  int pretrainMinutes = 60;
  /**
   * The Timeout minutes.
   */
  int timeoutMinutes = 60;
  /**
   * The Space training minutes.
   */
  int spaceTrainingMinutes = 120;
  /**
   * The Size.
   */
  int size = 256;
  
  @Override
  public void run(NotebookOutput log) {
    Tensor[][] trainingImages = getImages(log, size, 50, "kangaroo", "yin_yang");
    
    log.h1("First Layer");
    InitializationStep step0 = log.code(() -> {
      return new InitializationStep(log, trainingImages,
        size, pretrainMinutes, timeoutMinutes, 3, 7, 5);
    }).invoke();
    
    log.h1("Second Layer");
    AddLayerStep step1 = log.code(() -> {
      return new AddLayerStep(log, step0.trainingData, step0.model,
        2, step0.toSize, pretrainMinutes, timeoutMinutes,
        step0.band1, 11, 5, 2);
    }).invoke();
    
    log.h1("Third Layer");
    AddLayerStep step2 = log.code(() -> {
      return new AddLayerStep(log, step1.trainingData, step1.integrationModel,
        3, step1.toSize, pretrainMinutes, timeoutMinutes,
        step1.band2, 11, 5, 4);
    }).invoke();
  }
  
  @Override
  public HtmlNotebookOutput getLog() {
    try {
      String directoryName = new SimpleDateFormat("YYYY-MM-dd-HH-mm").format(new Date());
      File path = new File(Util.mkString(File.separator, "www", directoryName));
      path.mkdirs();
      File logFile = new File(path, "index.html");
      StreamNanoHTTPD server = new StreamNanoHTTPD(1999, "text/html", logFile).init();
      HtmlNotebookOutput log = new HtmlNotebookOutput(path, server.dataReciever);
      log.addCopy(rawOut);
      return log;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  protected FindFeatureSpace findFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
    return new FindFeatureSpace(log, features, inputBands) {
      @Override
      public FindFeatureSpace invoke() {
        ArrayList<Step> history = new ArrayList<>();
        TrainingMonitor monitor = getMonitor(history);
        int[] featureDimensions = features[0][1].getDimensions();
        FullyConnectedLayer synapseLayer = new FullyConnectedLayer(new int[]{inputBands}, featureDimensions);
        ImgBandBiasLayer bandBiasLayer = new ImgBandBiasLayer(featureDimensions[2]);
        
        PipelineNetwork network = log.code(() -> {
          PipelineNetwork pipelineNetwork = new PipelineNetwork(3);
          pipelineNetwork.add(synapseLayer);
          pipelineNetwork.add(bandBiasLayer);
          DAGNode sqLoss = pipelineNetwork.add(new MeanSqLossLayer(), pipelineNetwork.getHead(), pipelineNetwork.getInput(1));
          
          int[] categoryDimensions = features[0][0].getDimensions();
          pipelineNetwork.add(new FullyConnectedLayer(new int[]{inputBands}, categoryDimensions), pipelineNetwork.getInput(0));
          pipelineNetwork.add(new BiasLayer(categoryDimensions));
          pipelineNetwork.add(new SoftmaxActivationLayer());
          DAGNode entropy = pipelineNetwork.add(new EntropyLossLayer(), pipelineNetwork.getHead(), pipelineNetwork.getInput(2));
          
          pipelineNetwork.add(new SumInputsLayer(),
            pipelineNetwork.add(new LinearActivationLayer().freeze(), sqLoss),
            pipelineNetwork.add(new LinearActivationLayer().freeze(), entropy));
          return pipelineNetwork;
        });
        
        addPerformanceWrappers(log, network);
        log.p("Training feature network");
        Tensor[][] trainingData = log.code(() -> {
          return Arrays.stream(features).map(tensor -> new Tensor[]{
            new Tensor(inputBands).fill(() -> 1 * (FastRandom.random() - 0.5)), tensor[1], tensor[0]
          }).toArray(i -> new Tensor[i][]);
        });
        log.code(() -> {
          StochasticTrainable trainingSubject = new StochasticArrayTrainable(trainingData, network, trainingData.length / 5, trainingData.length);
          trainingSubject = (StochasticTrainable) ((TrainableDataMask) trainingSubject).setMask(true, false, false);
          ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, new ArrayTrainable(trainingData, network))
            .setMaxTrainingSize(trainingData.length)
            .setMinTrainingSize(1)
            .setMonitor(monitor)
            .setTimeout(spaceTrainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(1000);
          validatingTrainer.getRegimen().get(0)
            .setOrientation(new QuantifyOrientationWrapper(new QQN()))
            .setLineSearchFactory(name -> {
              if (name.contains("LBFGS") || name.contains("QQN")) {
                return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
              }
              else {
                return new ArmijoWolfeSearch().setMaxAlpha(1e6);
              }
            });
          validatingTrainer
            .run();
          
          averages = Arrays.copyOf(bandBiasLayer.getBias(), bandBiasLayer.getBias().length);
          vectors = IntStream.range(0, inputBands).mapToObj(inputBand -> {
            Tensor to = new Tensor(featureDimensions);
            to.fillByCoord(c -> synapseLayer.getWeights().get(inputBand, c.index));
            return to;
          }).toArray(i -> new Tensor[i]);
        });
        removePerformanceWrappers(log, network);
        printHistory(log, history);
        return this;
      }
    }.invoke();
  }
  
  /**
   * Add performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public void addPerformanceWrappers(NotebookOutput log, DAGNetwork network) {
    log.p("Adding performance wrappers");
    log.code(() -> {
      network.visitNodes(node -> {
        if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
          node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
        }
        else {
          ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
        }
      });
    });
  }
  
  /**
   * Remove performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public void removePerformanceWrappers(NotebookOutput log, DAGNetwork network) {
    log.p("Per-layer Performance Metrics:");
    log.code(() -> {
      Map<NNLayer, MonitoringWrapperLayer> metrics = new HashMap<>();
      network.visitNodes(node -> {
        if ((node.getLayer() instanceof MonitoringWrapperLayer)) {
          MonitoringWrapperLayer layer = node.getLayer();
          metrics.put(layer.getInner(), layer);
        }
      });
      System.out.println("Forward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        PercentileStatistics performance = e.getValue().getForwardPerformance();
        return String.format("%s -> %.4f +- %.4f (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
      }).reduce((a, b) -> a + "\n\t" + b));
      System.out.println("Backward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        PercentileStatistics performance = e.getValue().getBackwardPerformance();
        return String.format("%s -> %.4f +- %.4f (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
      }).reduce((a, b) -> a + "\n\t" + b));
    });
    log.p("Removing performance wrappers");
    log.code(() -> {
      network.visitNodes(node -> {
        if (node.getLayer() instanceof MonitoringWrapperLayer) {
          node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
        }
      });
    });
  }
  
  @Override
  protected void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, double factor_l1, boolean... mask) {
    if (network instanceof DAGNetwork) addPerformanceWrappers(log, (DAGNetwork) network);
    super.train(log, monitor, network, data, orientation, timeoutMinutes, factor_l1, mask);
    if (network instanceof DAGNetwork) removePerformanceWrappers(log, (DAGNetwork) network);
  }
}
