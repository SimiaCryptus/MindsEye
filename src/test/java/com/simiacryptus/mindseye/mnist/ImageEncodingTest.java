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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.media.ResampledSubLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.OwlQn;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class ImageEncodingTest extends ImageEncodingUtil {
  
  int displayImage = 2;
  
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != out) ((MarkdownNotebookOutput) log).addCopy(out);
      
      int pretrainMinutes = 1;
      int timeoutMinutes = 1;
      int size = 256;
      int images = 10;
      
      Tensor[][] trainingImages = getImages(log, size, 100, "kangaroo");
      
      log.h1("First Layer");
      InitializationStep step0 = log.code(()->{
        return new InitializationStep(log, trainingImages,
          images, size, pretrainMinutes, timeoutMinutes, 3, 7, 5);
      }).invoke();
      
      log.h1("Second Layer");
      AddLayerStep step1 = log.code(()->{
        return new AddLayerStep(log, step0.trainingData, step0.model,
          2, step0.toSize, pretrainMinutes, timeoutMinutes,
          step0.band1, 11, 5, 4);
      }).invoke();
      
      log.h1("Third Layer");
      AddLayerStep step2 = log.code(()->{
        return new AddLayerStep(log, step1.trainingData, step1.integrationModel,
          3, step1.toSize, pretrainMinutes, timeoutMinutes,
          step1.band2, 17, 5, 2);
      }).invoke();
      
      log.h1("Transcoding Different Category");
      TranscodeStep step3 = log.code(()->{
        return new TranscodeStep(log, "yin_yang",
          images, size, timeoutMinutes, step2.integrationModel, step2.toSize, step2.toSize, step2.band2);
      }).invoke();
      
    }
  }
  
  protected class TranscodeStep {
    public final int size;
    public final String category;
    public final int imageCount;
    public final NotebookOutput log;
    public final NNLayer model;
    public final Tensor[][] trainingData;
    public final TrainingMonitor monitor;
    public final int trainMinutes;
    public final List<Step> history = new ArrayList<>();
    
    public TranscodeStep(NotebookOutput log, String category, int imageCount, int size, int trainMinutes, NNLayer model, int... representationDims) {
      this.category = category;
      this.imageCount = imageCount;
      this.log = log;
      this.size = size;
      this.model = model;
      this.trainingData = addColumn(getImages(log, size, imageCount, category), representationDims);
      this.monitor = getMonitor(out, history);
      this.trainMinutes = trainMinutes;
    }
  
    @Override
    public String toString() {
      return "TranscodeStep{" +
        "category='" + category + '\'' +
        ", imageCount=" + imageCount +
        ", trainMinutes=" + trainMinutes +
        '}';
    }
  
    public TranscodeStep invoke() {
      log.h3("Training");
      DAGNetwork trainingModel0 = buildTrainingModel(log, model.copy().freeze(), 0, 1, 0, 0);
      train(log, monitor, trainingModel0, trainingData, new QQN(), trainMinutes, 0, false, true);
      printHistory(log, history);
      log.h3("Results");
      validationReport(log, trainingData, Arrays.asList(this.model), imageCount);
      printDataStatistics(log, trainingData);
      history.clear();
      return this;
    }
  }
  
  int modelNo = 0;
  public List<NNLayer> dataPipeline = new ArrayList<>();
  
  protected class InitializationStep {
    public final ConvolutionLayer convolutionLayer;
    public final ImgBandBiasLayer biasLayer;
    public final int fromSize;
    public final int toSize;
    public final NotebookOutput log;
    public final List<Step> history = new ArrayList<>();
    public final TrainingMonitor monitor;
    public final int pretrainMinutes;
    public final int timeoutMinutes;
    public final int images;
    public final int radius;
    public final DAGNetwork model;
    public final Tensor[][] trainingData;
    public final int band0;
    public final int band1;
    
    public InitializationStep(NotebookOutput log, Tensor[][] originalTrainingData, int images, int fromSize, int pretrainMinutes, int timeoutMinutes, int band0, int band1, int radius) {
      this.band1 = band1;
      this.band0 = band0;
      this.log = log;
      this.monitor = getMonitor(out, history);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.images = images;
      this.fromSize = fromSize;
      this.toSize = (fromSize + (radius - 1));
      this.trainingData = addColumn(originalTrainingData, toSize, toSize, band1);
      this.radius = radius;
      this.convolutionLayer = new ConvolutionLayer(radius, radius, band1, band0, false).setWeights(() -> 0.1 * (Math.random() - 0.5));
      this.biasLayer = new ImgBandBiasLayer(band0);
      this.model = buildModel();
    }
  
    @Override
    public String toString() {
      return "InitializationStep{" +
        ", fromSize=" + fromSize +
        ", toSize=" + toSize +
        ", pretrainMinutes=" + pretrainMinutes +
        ", timeoutMinutes=" + timeoutMinutes +
        ", images=" + images +
        ", radius=" + radius +
        ", band0=" + band0 +
        ", band1=" + band1 +
        '}';
    }
  
    public PipelineNetwork buildModel() {
      return log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(convolutionLayer);
        network.add(biasLayer);
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        return network;
      });
    }
    
    public InitializationStep invoke() {
      dataPipeline.add(model);
      initialize(log, convolutionFeatures(Arrays.stream(trainingData).map(x1 -> x1[0]), radius), convolutionLayer, biasLayer);
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, model.copy().freeze(), 0, 1, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, 0, false, true);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData, dataPipeline, displayImage);
        printModel(log, model, modelNo++);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Tuning");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, model, 0, 1, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new OwlQn(), timeoutMinutes, 0, false, true);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData, dataPipeline, displayImage);
        printModel(log, model, modelNo++);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      return this;
    }
    
  }
  
  protected class AddLayerStep {
    public final int toSize;
    public final ConvolutionLayer convolutionLayer;
    public final ImgBandBiasLayer biasLayer;
    public final PrintStream originalOut;
    public final NotebookOutput log;
    public final int layerNumber;
    public final int pretrainMinutes;
    public final int timeoutMinutes;
    public final int radius;
    public final int scale;
    public final List<Step> history;
    public final TrainingMonitor monitor;
    public final Tensor[][] trainingData;
    public final DAGNetwork innerModel;
    public final PipelineNetwork integrationModel;
    public final int band1;
    public final int band2;
    
    public AddLayerStep(NotebookOutput log, Tensor[][] trainingData, DAGNetwork priorModel, int layerNumber, int fromSize, int pretrainMinutes, int timeoutMinutes, int band1, int band2, int radius, int scale) {
      this.originalOut = out;
      this.log = log;
      this.band1 = band1;
      this.band2 = band2;
      this.layerNumber = layerNumber;
      this.scale = scale;
      if (0 != fromSize % scale) throw new IllegalArgumentException(fromSize + " % " + scale);
      this.toSize = (fromSize / scale + (radius - 1)) * scale; // 70
      this.trainingData = addColumn(trainingData, toSize, toSize, band2);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.radius = radius;
      this.history = new ArrayList<>();
      this.monitor = getMonitor(originalOut, history);
      this.convolutionLayer = new ConvolutionLayer(radius, radius, band2, band1, false).setWeights(() -> 0.01 * (Math.random() - 0.5));
      this.biasLayer = new ImgBandBiasLayer(band1);
      this.innerModel = buildNetwork();
      this.integrationModel = log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(innerModel);
        network.add(priorModel);
        return network;
      });
    }
  
    @Override
    public String toString() {
      return "AddLayerStep{" +
        "toSize=" + toSize +
        ", layerNumber=" + layerNumber +
        ", pretrainMinutes=" + pretrainMinutes +
        ", timeoutMinutes=" + timeoutMinutes +
        ", radius=" + radius +
        ", scale=" + scale +
        ", band1=" + band1 +
        ", band2=" + band2 +
        '}';
    }
  
    public AddLayerStep invoke() {
      dataPipeline.add(innerModel);
      Stream<Tensor> inputColumn = Arrays.stream(trainingData).map(x -> x[layerNumber - 1]);
      Tensor[] convolutionFeatures = convolutionFeatures(downExplodeTensors(inputColumn, scale), radius);
      initialize(log, convolutionFeatures, convolutionLayer, biasLayer);
      final boolean[] mask = getTrainingMask();
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel.copy().freeze(), layerNumber - 1, layerNumber, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData, dataPipeline, displayImage);
        printModel(log, innerModel, modelNo++);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Tuning");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel, layerNumber - 1, layerNumber, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), timeoutMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData, dataPipeline, displayImage);
        printModel(log, innerModel, modelNo++);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Integration Training");
        log.h3("Training");
        DAGNetwork trainingModel1 = buildTrainingModel(log, integrationModel, 0, layerNumber, 0, 0);
        train(log, monitor, trainingModel1, trainingData, new QQN(), timeoutMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData, dataPipeline, displayImage);
        printModel(log, innerModel, modelNo++);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      return this;
    }
    
    public boolean[] getTrainingMask() {
      final boolean[] mask = new boolean[layerNumber + 2];
      mask[layerNumber] = true;
      return mask;
    }
    
    public PipelineNetwork buildNetwork() {
      return log.code(() -> {
        return new PipelineNetwork(1,
          new ResampledSubLayer(scale,
            new PipelineNetwork(1,
              convolutionLayer,
              biasLayer)
          )
        );
      });
    }
    
    public PipelineNetwork getIntegrationModel() {
      return integrationModel;
    }
  }
}
