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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.eval.TrainableDataMask;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.ImgCropLayer;
import com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
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
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * The type Image encoding pca run.
 */
public class ImageEncodingPCATest {
  
  /**
   * The MnistProblemData pipeline.
   */
  public List<NNLayer> dataPipeline = new ArrayList<>();
  /**
   * The Display image.
   */
  int displayImage = 5;
  /**
   * The Model no.
   */
  int modelNo = 0;
  
  /**
   * Test.
   *
   * @throws Exception the exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    try (NotebookOutput log = report()) {
      run(log);
    }
  }
  
  /**
   * Gets log.
   *
   * @return the log
   */
  public NotebookOutput report() {
    MarkdownNotebookOutput log = MarkdownNotebookOutput.get(this);
    log.addCopy(TestUtil.rawOut);
    return log;
  }
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(NotebookOutput log) {
    int pretrainMinutes = 15;
    int timeoutMinutes = 30;
    int size = 256;
    
    Tensor[][] trainingImages = TestUtil.getImages(log, size, 5, "kangaroo");
    
    log.h1("First Layer");
    InitializationStep step0 = log.code(() -> {
      return new InitializationStep(log, trainingImages,
        size, pretrainMinutes, timeoutMinutes, 3, 12, 5);
    }).invoke(); // output: 260
    
    log.h1("Second Layer");
    AddLayerStep step1 = log.code(() -> {
      return new AddLayerStep(log, step0.trainingData, step0.model,
        2, step0.toSize, pretrainMinutes * 2, timeoutMinutes,
        step0.band1, 24, 5, 4);
    }).invoke(); // output: 274
    
    log.h1("Third Layer");
    AddLayerStep step2 = log.code(() -> {
      return new AddLayerStep(log, step1.trainingData, step1.integrationModel,
        3, step1.toSize, pretrainMinutes * 3, timeoutMinutes,
        step1.band2, 24, 5, 1);
    }).invoke(); // 276
  
    log.h1("Fourth Layer");
    AddLayerStep step3 = log.code(() -> {
      return new AddLayerStep(log, step2.trainingData, step2.integrationModel,
        4, step2.toSize, pretrainMinutes * 4, timeoutMinutes,
        step2.band2, 24, 5, 4);
    }).invoke(); // 278
  
    log.h1("Transcoding Different Category");
    TranscodeStep step4 = log.code(() -> {
      return new TranscodeStep(log, "yin_yang",
        100, size, timeoutMinutes*5, step3.integrationModel, step3.toSize, step3.toSize, step3.band2);
    }).invoke();
  }
  
  /**
   * The type Transcode runStep.
   */
  protected class TranscodeStep {
    /**
     * The Size.
     */
    public final int size;
    /**
     * The Category.
     */
    public final String category;
    /**
     * The Image count.
     */
    public final int imageCount;
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The Model.
     */
    public final NNLayer model;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
    /**
     * The Monitor.
     */
    public final TrainingMonitor monitor;
    /**
     * The Train minutes.
     */
    public final int trainMinutes;
    /**
     * The History.
     */
    public final List<Step> history = new ArrayList<>();
  
    /**
     * Instantiates a new Transcode runStep.
     *
     * @param log                the log
     * @param category           the category
     * @param imageCount         the image count
     * @param size               the size
     * @param trainMinutes       the train minutes
     * @param model              the model
     * @param representationDims the representation dims
     */
    public TranscodeStep(NotebookOutput log, String category, int imageCount, int size, int trainMinutes, NNLayer model, int... representationDims) {
      this.category = category;
      this.imageCount = imageCount;
      this.log = log;
      this.size = size;
      this.model = model;
      this.trainingData = TestUtil.addColumn(TestUtil.getImages(log, size, imageCount, category), representationDims);
      this.monitor = TestUtil.getMonitor(history);
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
  
    /**
     * Invoke transcode runStep.
     *
     * @return the transcode runStep
     */
    public TranscodeStep invoke() {
      log.h3("Training");
      DAGNetwork trainingModel0 = TestUtil.buildTrainingModel(model.copy().freeze(), 1, 2);
      train(log, monitor, trainingModel0, trainingData, new QQN(), trainMinutes, false, false, true);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      TestUtil.validationReport(log, trainingData, Arrays.asList(this.model), imageCount);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      return this;
    }
  }
  
  /**
   * The type Initialization runStep.
   */
  protected class InitializationStep {
    /**
     * The Convolution layer.
     */
    public final ConvolutionLayer convolutionLayer;
    /**
     * The Bias layer.
     */
    public final ImgBandBiasLayer biasLayer;
    /**
     * The From size.
     */
    public final int fromSize;
    /**
     * The To size.
     */
    public final int toSize;
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The History.
     */
    public final List<Step> history = new ArrayList<>();
    /**
     * The Monitor.
     */
    public final TrainingMonitor monitor;
    /**
     * The Pretrain minutes.
     */
    public final int pretrainMinutes;
    /**
     * The Timeout minutes.
     */
    public final int timeoutMinutes;
    /**
     * The Radius.
     */
    public final int radius;
    /**
     * The Model.
     */
    public final DAGNetwork model;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
    /**
     * The Band 0.
     */
    public final int band0;
    /**
     * The Band 1.
     */
    public final int band1;
  
    /**
     * Instantiates a new Initialization runStep.
     *
     * @param log                  the log
     * @param originalTrainingData the original training data
     * @param fromSize             the from size
     * @param pretrainMinutes      the pretrain minutes
     * @param timeoutMinutes       the timeout minutes
     * @param band0                the band 0
     * @param band1                the band 1
     * @param radius               the radius
     */
    public InitializationStep(NotebookOutput log, Tensor[][] originalTrainingData, int fromSize, int pretrainMinutes, int timeoutMinutes, int band0, int band1, int radius) {
      this.band1 = band1;
      this.band0 = band0;
      this.log = log;
      this.monitor = TestUtil.getMonitor(history);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.fromSize = fromSize;
      this.toSize = (fromSize + (radius - 1));
      this.trainingData = TestUtil.addColumn(originalTrainingData, toSize, toSize, band1);
      this.radius = radius;
      this.convolutionLayer = new ConvolutionLayer(radius, radius, band1, band0).setWeights(() -> 0.1 * (Math.random() - 0.5));
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
        ", radius=" + radius +
        ", band0=" + band0 +
        ", band1=" + band1 +
        '}';
    }
  
    /**
     * Build model pipeline network.
     *
     * @return the pipeline network
     */
    public PipelineNetwork buildModel() {
      return log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(convolutionLayer);
        network.add(biasLayer);
        network.add(new ImgCropLayer(fromSize, fromSize));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        //addLogging(network);
        return network;
      });
    }
  
    /**
     * Invoke initialization runStep.
     *
     * @return the initialization runStep
     */
    public InitializationStep invoke() {
      dataPipeline.add(model);
      log.code(() -> {
        initialize(log, TestUtil.convolutionFeatures(Arrays.stream(trainingData).map(x1 -> new Tensor[]{x1[0], x1[1]}), radius), convolutionLayer, biasLayer);
      });
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = TestUtil.buildTrainingModel(model.copy().freeze(), 1, 2);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, false, false, true);
        TestUtil.printHistory(log, history);
        log.h3("Results");
        TestUtil.validationReport(log, trainingData, dataPipeline, displayImage);
        TestUtil.printModel(log, model, modelNo++);
        TestUtil.printDataStatistics(log, trainingData);
        history.clear();
      }
      
      log.h2("Tuning");
      log.h3("Training");
      DAGNetwork trainingModel0 = TestUtil.buildTrainingModel(model, 1, 2);
      train(log, monitor, trainingModel0, trainingData, new OwlQn(), timeoutMinutes, false, false, true);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      TestUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      TestUtil.printModel(log, model, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      
      return this;
    }
    
  }
  
  /**
   * The type Add layer runStep.
   */
  protected class AddLayerStep {
    /**
     * The To size.
     */
    public final int toSize;
    /**
     * The Convolution layer.
     */
    public final ConvolutionLayer convolutionLayer;
    /**
     * The Bias layer.
     */
    public final ImgBandBiasLayer biasLayer;
    /**
     * The Original out.
     */
    public final PrintStream originalOut;
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The Layer number.
     */
    public final int layerNumber;
    /**
     * The Pretrain minutes.
     */
    public final int pretrainMinutes;
    /**
     * The Timeout minutes.
     */
    public final int timeoutMinutes;
    /**
     * The Radius.
     */
    public final int radius;
    /**
     * The Scale.
     */
    public final int scale;
    /**
     * The History.
     */
    public final List<Step> history;
    /**
     * The Monitor.
     */
    public final TrainingMonitor monitor;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
    /**
     * The Inner model.
     */
    public final DAGNetwork innerModel;
    /**
     * The Integration model.
     */
    public final PipelineNetwork integrationModel;
    /**
     * The Band 1.
     */
    public final int band1;
    /**
     * The Band 2.
     */
    public final int band2;
    private final int fromSize;
  
    /**
     * Instantiates a new Add layer runStep.
     *
     * @param log             the log
     * @param trainingData    the training data
     * @param priorModel      the prior model
     * @param layerNumber     the layer number
     * @param fromSize        the from size
     * @param pretrainMinutes the pretrain minutes
     * @param timeoutMinutes  the timeout minutes
     * @param band1           the band 1
     * @param band2           the band 2
     * @param radius          the radius
     * @param scale           the scale
     */
    public AddLayerStep(NotebookOutput log, Tensor[][] trainingData, DAGNetwork priorModel, int layerNumber, int fromSize, int pretrainMinutes, int timeoutMinutes, int band1, int band2, int radius, int scale) {
      this.originalOut = TestUtil.rawOut;
      this.log = log;
      this.band1 = band1;
      this.band2 = band2;
      this.layerNumber = layerNumber;
      this.scale = scale;
      if (0 != fromSize % scale) throw new IllegalArgumentException(fromSize + " % " + scale);
      this.fromSize = fromSize;
      this.toSize = (fromSize / scale + (radius - 1)) * scale; // 70
      this.trainingData = TestUtil.addColumn(trainingData, toSize, toSize, band2);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.radius = radius;
      this.history = new ArrayList<>();
      this.monitor = TestUtil.getMonitor(history);
      this.convolutionLayer = new ConvolutionLayer(radius, radius, band2, band1).setWeights(() -> 0.01 * (Math.random() - 0.5));
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
  
    /**
     * Invoke add layer runStep.
     *
     * @return the add layer runStep
     */
    public AddLayerStep invoke() {
      dataPipeline.add(innerModel);
      Stream<Tensor[]> inputColumn = Arrays.stream(trainingData).map(x -> new Tensor[]{x[0], x[layerNumber]});
      Tensor[][] convolutionFeatures = TestUtil.convolutionFeatures(TestUtil.downExplodeTensors(inputColumn, scale), radius);
      log.code(() -> {
        initialize(log, convolutionFeatures, convolutionLayer, biasLayer);
      });
      final boolean[] mask = getTrainingMask();
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = TestUtil.buildTrainingModel(innerModel.copy().freeze(), layerNumber, layerNumber + 1);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, mask);
        TestUtil.printHistory(log, history);
        log.h3("Results");
        TestUtil.validationReport(log, trainingData, dataPipeline, displayImage);
        TestUtil.printModel(log, innerModel, modelNo++);
        TestUtil.printDataStatistics(log, trainingData);
        history.clear();
      }
      
      log.h2("Tuning");
      log.h3("Training");
      DAGNetwork trainingModel0 = TestUtil.buildTrainingModel(innerModel, layerNumber, layerNumber + 1);
      train(log, monitor, trainingModel0, trainingData, new QQN(), timeoutMinutes, mask);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      TestUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      TestUtil.printModel(log, innerModel, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      
      log.h2("Integration Training");
      log.h3("Training");
      DAGNetwork trainingModel1 = TestUtil.buildTrainingModel(integrationModel, 1, layerNumber + 1);
      train(log, monitor, trainingModel1, trainingData, new QQN(), timeoutMinutes, mask);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      TestUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      TestUtil.printModel(log, innerModel, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      return this;
    }
  
    /**
     * Get training mask boolean [ ].
     *
     * @return the boolean [ ]
     */
    public boolean[] getTrainingMask() {
      final boolean[] mask = new boolean[layerNumber + 3];
      mask[layerNumber + 1] = true;
      return mask;
    }
  
    /**
     * Build network pipeline network.
     *
     * @return the pipeline network
     */
    public PipelineNetwork buildNetwork() {
      return log.code(() -> {
        return new PipelineNetwork(1,
          new RescaledSubnetLayer(scale,
            new PipelineNetwork(1,
              convolutionLayer,
              biasLayer
            )
          ), new ImgCropLayer(fromSize, fromSize)
        );
      });
    }
  
    /**
     * Gets integration model.
     *
     * @return the integration model
     */
    public PipelineNetwork getIntegrationModel() {
      return integrationModel;
    }
  }
  
  /**
   * Initialize.
   *
   * @param log              the log
   * @param features         the features
   * @param convolutionLayer the convolution layer
   * @param biasLayer        the bias layer
   */
  protected void initialize(NotebookOutput log, Tensor[][] features, ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer) {
    Tensor prototype = features[0][1];
    int[] dimensions = prototype.getDimensions();
    int[] filterDimensions = convolutionLayer.kernel.getDimensions();
    assert filterDimensions[0] == dimensions[0];
    assert filterDimensions[1] == dimensions[1];
    int outputBands = dimensions[2];
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    FindFeatureSpace findFeatureSpace = findFeatureSpace(log, features, inputBands);
    TestUtil.setInitialFeatureSpace(convolutionLayer, biasLayer, findFeatureSpace);
  }
  
  /**
   * Find feature space find feature space.
   *
   * @param log        the log
   * @param features   the features
   * @param inputBands the input bands
   * @return the find feature space
   */
  protected FindFeatureSpace findFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
    return new FindPCAFeatures(log, inputBands, features).invoke();
  }
  
  /**
   * Train.
   *
   * @param log            the log
   * @param monitor        the monitor
   * @param network        the network
   * @param data           the data
   * @param orientation    the orientation
   * @param timeoutMinutes the timeout minutes
   * @param mask           the mask
   */
  protected void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, boolean... mask) {
    log.out("Training for %s minutes, mask=%s", timeoutMinutes, Arrays.toString(mask));
    log.code(() -> {
      SampledTrainable trainingSubject = new SampledArrayTrainable(data, network, data.length);
      trainingSubject = (SampledTrainable) ((TrainableDataMask) trainingSubject).setMask(mask);
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, new ArrayTrainable(data, network))
        .setMaxTrainingSize(data.length)
        .setMinTrainingSize(1)
        .setMonitor(monitor)
        .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
        .setMaxIterations(1000);
      validatingTrainer.getRegimen().get(0)
        .setOrientation(orientation)
        .setLineSearchFactory(name -> new QuadraticSearch().setCurrentRate(1.0));
      validatingTrainer
        .run();
    });
  }
  
}
