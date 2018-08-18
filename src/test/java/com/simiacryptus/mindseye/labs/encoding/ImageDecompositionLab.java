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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledTrainable;
import com.simiacryptus.mindseye.eval.TrainableDataMask;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.ImgCropLayer;
import com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * The type Image encoding pca apply.
 */
public class ImageDecompositionLab {
  
  /**
   * The MnistProblemData pipeline.
   */
  @Nonnull
  public List<Layer> dataPipeline = new ArrayList<>();
  /**
   * The Display png.
   */
  int displayImage = 5;
  /**
   * The Model no.
   */
  int modelNo = 0;
  
  /**
   * Test.
   *
   * @param args the input arguments
   * @throws Exception the exception
   */
  public static void main(final CharSequence... args) throws Exception {
    @Nonnull final ImageDecompositionLab lab = new ImageDecompositionLab();
    try (@Nonnull NotebookOutput log = lab.report()) {
      lab.run(log);
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
  protected void initialize(final NotebookOutput log, @Nonnull final Supplier<Stream<Tensor[]>> features, @Nonnull final ConvolutionLayer convolutionLayer, @Nonnull final ImgBandBiasLayer biasLayer) {
    final Tensor prototype = features.get().findAny().get()[1];
    @Nonnull final int[] dimensions = prototype.getDimensions();
    @Nonnull final int[] filterDimensions = convolutionLayer.getKernel().getDimensions();
    assert filterDimensions[0] == dimensions[0];
    assert filterDimensions[1] == dimensions[1];
    final int outputBands = dimensions[2];
    if (outputBands != biasLayer.getBias().length) {
      throw new AssertionError(String.format("%d != %d", outputBands, biasLayer.getBias().length));
    }
    final int inputBands = filterDimensions[2] / outputBands;
    @Nonnull final FindFeatureSpace findFeatureSpace = new FindPCAFeatures(log, inputBands) {
      @Override
      public Stream<Tensor[]> getFeatures() {
        return features.get();
      }
    }.invoke();
    EncodingUtil.setInitialFeatureSpace(convolutionLayer, biasLayer, findFeatureSpace);
  }
  
  /**
   * Gets log.
   *
   * @return the log
   */
  @Nonnull
  public NotebookOutput report() {
    try {
      @Nonnull final CharSequence directoryName = new SimpleDateFormat("YYYY-MM-dd-HH-mm").format(new Date());
      @Nonnull final File path = new File(Util.mkString(File.separator, "www", directoryName));
      path.mkdirs();
      @Nonnull final NotebookOutput log = new MarkdownNotebookOutput(path, getClass().getSimpleName(), true);
      return log;
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(@Nonnull final NotebookOutput log) {
    final int pretrainMinutes = 30;
    final int timeoutMinutes = 30;
    final int images = 10;
    final int size = 400;
    @Nonnull String source = "H:\\SimiaCryptus\\photos";
    displayImage = images;
  
    final Tensor[][] trainingImages = null == source ? EncodingUtil.getImages(log, size, images, "kangaroo") :
      Arrays.stream(new File(source).listFiles()).map(input -> {
        try {
          return ImageIO.read(input);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).map(img -> new Tensor[]{
        new Tensor(1.0),
        Tensor.fromRGB(TestUtil.resize(img, size))
      }).toArray(i -> new Tensor[i][]);
  
    Arrays.stream(trainingImages).map(x -> x[1]).map(x -> x.toImage()).map(x -> {
      return log.png(x, "example");
    }).forEach(str -> log.p(str));
  
    log.h1("First LayerBase");
    @Nonnull final InitializationStep step0 = log.eval(() -> {
      return new InitializationStep(log, trainingImages,
        size, pretrainMinutes, timeoutMinutes, 3, 9, 5);
    }).invoke(); // output: 260
  
    log.h1("Second LayerBase");
    @Nonnull final AddLayerStep step1 = log.eval(() -> {
      return new AddLayerStep(log, step0.trainingData, step0.model,
        2, step0.toSize, pretrainMinutes * 2, timeoutMinutes,
        step0.band1, 18, 3, 4);
    }).invoke(); // output: 274
  
    log.h1("Third LayerBase");
    @Nonnull final AddLayerStep step2 = log.eval(() -> {
      return new AddLayerStep(log, step1.trainingData, step1.integrationModel,
        3, step1.toSize, pretrainMinutes * 3, timeoutMinutes,
        step1.band2, 48, 3, 1);
    }).invoke(); // 276
  
    log.h1("Fourth LayerBase");
    @Nonnull final AddLayerStep step3 = log.eval(() -> {
      return new AddLayerStep(log, step2.trainingData, step2.integrationModel,
        4, step2.toSize, pretrainMinutes * 4, timeoutMinutes,
        step2.band2, 48, 5, 4);
    }).invoke(); // 278
    
    log.h1("Transcoding Different Category");
    log.eval(() -> {
      return new TranscodeStep(log, "yin_yang",
        images, size, timeoutMinutes * 5, step3.integrationModel, step3.toSize, step3.toSize, step3.band2);
    }).invoke();
  }
  
  /**
   * Train.
   *
   * @param log            the log
   * @param monitor        the monitor
   * @param network        the network
   * @param data           the data
   * @param timeoutMinutes the timeout minutes
   * @param mask           the mask
   */
  protected void train(@Nonnull final NotebookOutput log, final TrainingMonitor monitor, final Layer network, @Nonnull final Tensor[][] data, final int timeoutMinutes, final boolean... mask) {
    log.out("Training for %s minutes, mask=%s", timeoutMinutes, Arrays.toString(mask));
    log.run(() -> {
      @Nonnull SampledTrainable trainingSubject = new SampledArrayTrainable(data, network, data.length);
      trainingSubject = (SampledTrainable) ((TrainableDataMask) trainingSubject).setMask(mask);
      @Nonnull final ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, new ArrayTrainable(data, network))
        .setMaxTrainingSize(data.length)
        .setMinTrainingSize(5)
        .setMonitor(monitor)
        .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
        .setMaxIterations(1000);
      validatingTrainer.getRegimen().get(0)
        .setOrientation(new GradientDescent())
        .setLineSearchFactory(name -> name.equals(QQN.CURSOR_NAME) ?
          new QuadraticSearch().setCurrentRate(1.0) :
          new QuadraticSearch().setCurrentRate(1.0));
      validatingTrainer
        .run();
    });
  }
  
  /**
   * The type Add layer runStep.
   */
  protected class AddLayerStep {
    /**
     * The Band 1.
     */
    public final int band1;
    /**
     * The Band 2.
     */
    public final int band2;
    /**
     * The Bias layer.
     */
    @Nonnull
    public final ImgBandBiasLayer biasLayer;
    /**
     * The Convolution layer.
     */
    @Nonnull
    public final ConvolutionLayer convolutionLayer;
    /**
     * The History.
     */
    @Nonnull
    public final List<StepRecord> history;
    /**
     * The Inner model.
     */
    public final DAGNetwork innerModel;
    /**
     * The Integration model.
     */
    public final PipelineNetwork integrationModel;
    /**
     * The LayerBase number.
     */
    public final int layerNumber;
    /**
     * The Log.
     */
    @Nonnull
    public final NotebookOutput log;
    /**
     * The Monitor.
     */
    @Nonnull
    public final TrainingMonitor monitor;
    /**
     * The Original out.
     */
    @Nonnull
    public final PrintStream originalOut;
    /**
     * The Pretrain minutes.
     */
    public final int pretrainMinutes;
    /**
     * The Radius.
     */
    public final int radius;
    /**
     * The Scale.
     */
    public final int scale;
    /**
     * The Timeout minutes.
     */
    public final int timeoutMinutes;
    /**
     * The To size.
     */
    public final int toSize;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
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
    public AddLayerStep(@Nonnull final NotebookOutput log, @Nonnull final Tensor[][] trainingData, final DAGNetwork priorModel,
      final int layerNumber, final int fromSize, final int pretrainMinutes, final int timeoutMinutes,
      final int band1, final int band2, final int radius, final int scale) {
      originalOut = EncodingUtil.rawOut;
      this.log = log;
      this.band1 = band1;
      this.band2 = band2;
      this.layerNumber = layerNumber;
      this.scale = scale;
      if (0 != fromSize % scale) throw new IllegalArgumentException(fromSize + " % " + scale);
      this.fromSize = fromSize;
      toSize = (fromSize / scale + radius - 1) * scale; // 70
      Arrays.stream(trainingData).allMatch(x -> x.length == this.layerNumber - 1);
      this.trainingData = EncodingUtil.addColumn(trainingData, toSize, toSize, band2);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.radius = radius;
      history = new ArrayList<>();
      monitor = EncodingUtil.getMonitor(history);
      convolutionLayer = new ConvolutionLayer(radius, radius, band2, band1).set(i -> 0.01 * (Math.random() - 0.5));
      biasLayer = new ImgBandBiasLayer(band1);
      innerModel = buildNetwork();
      integrationModel = log.eval(() -> {
        @Nonnull final PipelineNetwork network = new PipelineNetwork(1);
        network.add(innerModel).freeRef();
        network.add(priorModel).freeRef();
        return network;
      });
    }
  
    /**
     * Build network pipeline network.
     *
     * @return the pipeline network
     */
    public PipelineNetwork buildNetwork() {
      return log.eval(() -> {
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
  
    /**
     * Get training mask boolean [ ].
     *
     * @return the boolean [ ]
     */
    @Nonnull
    public boolean[] getTrainingMask() {
      @Nonnull final boolean[] mask = new boolean[layerNumber + 2];
      mask[layerNumber + 1] = true;
      return mask;
    }
  
    /**
     * Invoke add layer runStep.
     *
     * @return the add layer runStep
     */
    @Nonnull
    public AddLayerStep invoke() {
      dataPipeline.add(innerModel);
      log.run(() -> {
        initialize(log, () -> {
          @Nonnull final Stream<Tensor[]> tensors = EncodingUtil.downExplodeTensors(Arrays.stream(trainingData).map(x -> new Tensor[]{x[0], x[layerNumber]}), scale);
          return EncodingUtil.convolutionFeatures(tensors, radius, 1);
        }, convolutionLayer, biasLayer);
      });
      @Nonnull final boolean[] mask = getTrainingMask();
      
      {
        log.h2("Initialization");
        log.h3("Training");
        @Nonnull final DAGNetwork trainingModel0 = EncodingUtil.buildTrainingModel(innerModel.copy().freeze(), layerNumber, layerNumber + 1);
        train(log, monitor, trainingModel0, trainingData, pretrainMinutes, mask);
        TestUtil.printHistory(log, history);
        log.h3("Results");
        EncodingUtil.validationReport(log, trainingData, dataPipeline, displayImage);
        EncodingUtil.printModel(log, innerModel, modelNo++);
        TestUtil.printDataStatistics(log, trainingData);
        history.clear();
      }
  
      log.h2("Tuning");
      log.h3("Training");
      @Nonnull final DAGNetwork trainingModel0 = EncodingUtil.buildTrainingModel(innerModel, layerNumber, layerNumber + 1);
      train(log, monitor, trainingModel0, trainingData, timeoutMinutes, mask);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      EncodingUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      EncodingUtil.printModel(log, innerModel, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
  
      log.h2("Integration Training");
      log.h3("Training");
      @Nonnull final DAGNetwork trainingModel1 = EncodingUtil.buildTrainingModel(integrationModel, 1, layerNumber + 1);
      train(log, monitor, trainingModel1, trainingData, timeoutMinutes, mask);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      EncodingUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      EncodingUtil.printModel(log, innerModel, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      return this;
    }
  
    @Nonnull
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
  }
  
  /**
   * The type Initialization runStep.
   */
  protected class InitializationStep {
    /**
     * The Band 0.
     */
    public final int band0;
    /**
     * The Band 1.
     */
    public final int band1;
    /**
     * The Bias layer.
     */
    @Nonnull
    public final ImgBandBiasLayer biasLayer;
    /**
     * The Convolution layer.
     */
    @Nonnull
    public final ConvolutionLayer convolutionLayer;
    /**
     * The From size.
     */
    public final int fromSize;
    /**
     * The History.
     */
    public final List<StepRecord> history = new ArrayList<>();
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The Model.
     */
    public final DAGNetwork model;
    /**
     * The Monitor.
     */
    @Nonnull
    public final TrainingMonitor monitor;
    /**
     * The Pretrain minutes.
     */
    public final int pretrainMinutes;
    /**
     * The Radius.
     */
    public final int radius;
    /**
     * The Timeout minutes.
     */
    public final int timeoutMinutes;
    /**
     * The To size.
     */
    public final int toSize;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
  
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
    public InitializationStep(final NotebookOutput log, @Nonnull final Tensor[][] originalTrainingData, final int fromSize, final int pretrainMinutes, final int timeoutMinutes, final int band0, final int band1, final int radius) {
      this.band1 = band1;
      this.band0 = band0;
      this.log = log;
      monitor = EncodingUtil.getMonitor(history);
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.fromSize = fromSize;
      toSize = fromSize + radius - 1;
      trainingData = EncodingUtil.addColumn(originalTrainingData, toSize, toSize, band1);
      this.radius = radius;
      convolutionLayer = new ConvolutionLayer(radius, radius, band1, band0).set(i -> 0.1 * (Math.random() - 0.5));
      biasLayer = new ImgBandBiasLayer(band0);
      model = buildModel();
    }
  
    /**
     * Build model pipeline network.
     *
     * @return the pipeline network
     */
    public PipelineNetwork buildModel() {
      return log.eval(() -> {
        @Nonnull final PipelineNetwork network = new PipelineNetwork(1);
        network.add(convolutionLayer).freeRef();
        network.add(biasLayer).freeRef();
        network.wrap(new ImgCropLayer(fromSize, fromSize)).freeRef();
        network.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
        //addLogging(network);
        return network;
      });
    }
  
    /**
     * Invoke initialization runStep.
     *
     * @return the initialization runStep
     */
    @Nonnull
    public InitializationStep invoke() {
      dataPipeline.add(model);
      log.run(() -> {
        initialize(log, () -> EncodingUtil.convolutionFeatures(Arrays.stream(trainingData).map(x1 -> new Tensor[]{x1[0], x1[1]}), radius, 1), convolutionLayer, biasLayer);
      });
      
      {
        log.h2("Initialization");
        log.h3("Training");
        @Nonnull final DAGNetwork trainingModel0 = EncodingUtil.buildTrainingModel(model.copy().freeze(), 1, 2);
        train(log, monitor, trainingModel0, trainingData, pretrainMinutes, false, false, true);
        TestUtil.printHistory(log, history);
        log.h3("Results");
        EncodingUtil.validationReport(log, trainingData, dataPipeline, displayImage);
        EncodingUtil.printModel(log, model, modelNo++);
        TestUtil.printDataStatistics(log, trainingData);
        history.clear();
      }
      
      log.h2("Tuning");
      log.h3("Training");
      @Nonnull final DAGNetwork trainingModel0 = EncodingUtil.buildTrainingModel(model, 1, 2);
      train(log, monitor, trainingModel0, trainingData, timeoutMinutes, false, false, true);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      EncodingUtil.validationReport(log, trainingData, dataPipeline, displayImage);
      EncodingUtil.printModel(log, model, modelNo++);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      
      return this;
    }
  
    @Nonnull
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
    
  }
  
  /**
   * The type Transcode runStep.
   */
  protected class TranscodeStep {
    /**
     * The Category.
     */
    public final String category;
    /**
     * The History.
     */
    public final List<StepRecord> history = new ArrayList<>();
    /**
     * The Image count.
     */
    public final int imageCount;
    /**
     * The Log.
     */
    @Nonnull
    public final NotebookOutput log;
    /**
     * The Model.
     */
    public final Layer model;
    /**
     * The Monitor.
     */
    @Nonnull
    public final TrainingMonitor monitor;
    /**
     * The Size.
     */
    public final int size;
    /**
     * The Training data.
     */
    public final Tensor[][] trainingData;
    /**
     * The Train minutes.
     */
    public final int trainMinutes;
  
    /**
     * Instantiates a new Transcode runStep.
     *
     * @param log                the log
     * @param category           the category
     * @param imageCount         the png count
     * @param size               the size
     * @param trainMinutes       the trainCjGD minutes
     * @param model              the model
     * @param representationDims the representation dims
     */
    public TranscodeStep(@Nonnull final NotebookOutput log, final String category, final int imageCount, final int size, final int trainMinutes, final Layer model, final int... representationDims) {
      this.category = category;
      this.imageCount = imageCount;
      this.log = log;
      this.size = size;
      this.model = model;
      trainingData = EncodingUtil.addColumn(EncodingUtil.getImages(log, size, imageCount, category), representationDims);
      monitor = EncodingUtil.getMonitor(history);
      this.trainMinutes = trainMinutes;
    }
  
    /**
     * Invoke transcode runStep.
     *
     * @return the transcode runStep
     */
    @Nonnull
    public TranscodeStep invoke() {
      log.h3("Training");
      @Nonnull final DAGNetwork trainingModel0 = EncodingUtil.buildTrainingModel(model.copy().freeze(), 1, 2);
      train(log, monitor, trainingModel0, trainingData, trainMinutes, false, false, true);
      TestUtil.printHistory(log, history);
      log.h3("Results");
      EncodingUtil.validationReport(log, trainingData, Arrays.asList(model), imageCount);
      TestUtil.printDataStatistics(log, trainingData);
      history.clear();
      return this;
    }
  
    @Nonnull
    @Override
    public String toString() {
      return "TranscodeStep{" +
        "category='" + category + '\'' +
        ", imageCount=" + imageCount +
        ", trainMinutes=" + trainMinutes +
        '}';
    }
  }
  
}
