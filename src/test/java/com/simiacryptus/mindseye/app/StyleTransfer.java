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

package com.simiacryptus.mindseye.app;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.labs.encoding.PCAUtil;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.GateBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 */
public class StyleTransfer extends ArtistryAppBase {
  
  
  /**
   * The Texture netork.
   */
  int imageSize = 600;
  
  /**
   * With clamp pipeline network.
   *
   * @param network1 the network 1
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork withClamp(final PipelineNetwork network1) {
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(getClamp(255));
    network.wrap(network1);
    return network;
  }
  
  /**
   * Sets precision.
   *
   * @param network   the network
   * @param precision the precision
   */
  public static void setPrecision(final DAGNetwork network, final Precision precision) {
    network.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }
  
  /**
   * Pca tensor.
   *
   * @param cov   the cov
   * @param power the power
   * @return the tensor
   */
  @Nonnull
  public static Tensor pca(final Tensor cov, final double power) {
    final int inputbands = (int) Math.sqrt(cov.getDimensions()[2]);
    final int outputbands = inputbands;
    Array2DRowRealMatrix realMatrix = new Array2DRowRealMatrix(inputbands, inputbands);
    cov.coordStream(false).forEach(c -> {
      double v = cov.get(c);
      int x = c.getIndex() % inputbands;
      int y = (c.getIndex() - x) / inputbands;
      realMatrix.setEntry(x, y, v);
    });
    Tensor[] features = PCAUtil.pcaFeatures(realMatrix, outputbands, new int[]{1, 1, inputbands}, power);
    Tensor kernel = new Tensor(1, 1, inputbands * outputbands);
    PCAUtil.populatePCAKernel_1(kernel, features);
    return kernel;
  }
  
  /**
   * Randomize buffered image.
   *
   * @param contentImage the content image
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage randomize(final BufferedImage contentImage) {
    return Tensor.fromRGB(contentImage).map(x -> FastRandom.INSTANCE.random() * 100).toRgbImage();
  }
  
  /**
   * Gets clamp.
   *
   * @param max the max
   * @return the clamp
   */
  @Nonnull
  public static PipelineNetwork getClamp(final int max) {
    @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(max).setScale(-1).freeze());
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(max).setScale(-1).freeze());
    return clamp;
  }
  
  /**
   * To json string.
   *
   * @param obj the style parameters
   * @return the string
   */
  public static String toJson(final Object obj) {
    String json;
    try {
      ObjectMapper mapper = new ObjectMapper();
      mapper.configure(SerializationFeature.INDENT_OUTPUT, true);
      json = mapper.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
    return json;
  }
  
  /**
   * Load buffered image.
   *
   * @param image     the style
   * @param imageSize the image size
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage load(final String image, final int imageSize) {
    BufferedImage bufferedImage;
    try {
      bufferedImage = ImageIO.read(new File(image));
      bufferedImage = TestUtil.resize(bufferedImage, imageSize, true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return bufferedImage;
  }
  
  /**
   * Load buffered image.
   *
   * @param imageFile the style
   * @param width     the width
   * @param height    the height
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage load(final String imageFile, final int width, final int height) {
    BufferedImage image;
    try {
      image = ImageIO.read(new File(imageFile));
      image = TestUtil.resize(image, width, height);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return image;
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run, "StyleTransfer_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup styleParameters, final int trainingMinutes) {
    NeuralSetup neuralSetup = new NeuralSetup(log, styleParameters).init();
    PipelineNetwork network = neuralSetup.fitnessNetwork(log);
    log.p("Input Parameters:");
    log.code(() -> {
      return toJson(styleParameters);
    });
    try {
      log.p("Input Content:");
      log.p(log.image(styleParameters.contentImage, "Content Image"));
      log.p("Style Content:");
      styleParameters.styleImages.forEach((file, styleImage) -> {
        try {
          log.p(log.image(styleImage, file));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      log.p("Input Canvas:");
      log.p(log.image(canvasImage, "Input Canvas"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    BufferedImage result = train(log, canvasImage, network, styleParameters.precision, trainingMinutes);
    try {
      log.p("Output Canvas:");
      log.p(log.image(result, "Output Canvas"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_0(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase0() {
          super.phase0();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.setLarge(true).getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_1a(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase1a() {
          super.phase1a();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.setLarge(true).getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_1b(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase1b() {
          super.phase1b();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.setLarge(true).getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_1c(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase1c() {
          super.phase1c();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.setLarge(true).getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_1d(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase1d() {
          super.phase1d();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_1e(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase1e() {
          super.phase1e();
          layers[0] = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.getNetwork();
    } catch (@Nonnull final RuntimeException e) {
    } catch (Throwable e) {
      throw new RuntimeException(e);
    }
    return layers[0];
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @param mean    the mean
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork gram(final PipelineNetwork network, Tensor mean) {
    network.wrap(new ImgBandBiasLayer(mean.scale(-1)));
    network.wrap(new GramianLayer());
    return network;
  }
  
  /**
   * Load list.
   *
   * @param style     the style
   * @param imageSize the image size
   * @return the list
   */
  @Nonnull
  public List<BufferedImage> load(final List<String> style, final int imageSize) {
    return style.stream().map(x -> load(x, imageSize)).collect(Collectors.toList());
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init(log);
    Precision precision = Precision.Float;
    imageSize = 400;
    double growthFactor = Math.sqrt(1.5);
    String lakeAndForest = "H:\\SimiaCryptus\\ArtistryAppBase\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    String vanGogh = "H:\\SimiaCryptus\\ArtistryAppBase\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    String threeMusicians = "H:\\SimiaCryptus\\ArtistryAppBase\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  
    Map<String, StyleCoefficients> styles = new HashMap<>();
    double contentCoeff = 1e5;
    styles.put(lakeAndForest, new StyleCoefficients(
      contentCoeff * 1e-1, contentCoeff * 1e-1,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0, false));
    styles.put(threeMusicians, new StyleCoefficients(
      1e-3, 1e-3,
      1e-3, 1e-3,
      1e-2, 1e-5,
      1e-2, 1e-5,
      1e-2, 1e-5,
      1e-2, 1e-5, false)
    );
    ContentCoefficients contentCoefficients = new ContentCoefficients(
      contentCoeff * 1e-2,
      contentCoeff * 1e-2,
      contentCoeff * 1e-2,
      contentCoeff * 1e-2,
      contentCoeff * 1e-2,
      contentCoeff * 1e-2);
    double power = 0.0;
    int trainingMinutes = 90;
  
    log.h1("Phase 0");
    BufferedImage canvasImage = load(lakeAndForest, imageSize);
    canvasImage = randomize(canvasImage);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    Map<String, BufferedImage> styleImages = new HashMap<>();
    styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (imageSize * 1.5)) : imageSize)));
    BufferedImage contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
    canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
    for (int i = 1; i < 10; i++) {
      log.h1("Phase " + i);
      imageSize = (int) (imageSize * growthFactor);
      styleImages.clear();
      styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (imageSize * 1.5)) : imageSize)));
      canvasImage = TestUtil.resize(canvasImage, imageSize, true);
      contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
      canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Avg pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork avg(final PipelineNetwork network) {
    network.wrap(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg));
    return network;
  }
  
  /**
   * Train buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param network         the network
   * @param precision       the precision
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage train(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final PipelineNetwork network, final Precision precision, final int trainingMinutes) {
    System.gc();
    Tensor canvas = Tensor.fromRGB(canvasImage);
    TestUtil.monitorImage(canvas, false, false);
    network.setFrozen(true);
    setPrecision(network, precision);
    @Nonnull Trainable trainable = new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    TestUtil.instrumentPerformance(log, network);
    addLayersHandler(network, server);
  
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history))
        .setOrientation(new QQN())
        //.setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(trainingMinutes, TimeUnit.MINUTES)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
      return TestUtil.plot(history);
    });
    return canvas.toImage();
  }
  
  /**
   * Gets training monitor.
   *
   * @param history the history
   * @return the training monitor
   */
  @Nonnull
  public TrainingMonitor getTrainingMonitor(@Nonnull ArrayList<StepRecord> history) {
    return TestUtil.getMonitor(history);
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network      the network
   * @param mean         the mean
   * @param pcaTransform the pca transform
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork gram(final PipelineNetwork network, Tensor mean, Tensor pcaTransform) {
    int[] dimensions = pcaTransform.getDimensions();
    int inputBands = mean.getDimensions()[2];
    int pcaBands = dimensions[2];
    int outputBands = pcaBands / inputBands;
    int width = dimensions[0];
    int height = dimensions[1];
    network.wrap(new ImgBandBiasLayer(mean.scale(-1)));
    network.wrap(new ConvolutionLayer(width, height, inputBands, outputBands).set(pcaTransform));
    network.wrap(new GramianLayer());
    return network;
  }
  
  /**
   * The type Content coefficients.
   */
  public static class ContentCoefficients {
    /**
     * The Coeff content 0.
     */
    public final double coeff_content_0;
    /**
     * The Coeff content 1 a.
     */
    public final double coeff_content_1a;
    /**
     * The Coeff content 1 b.
     */
    public final double coeff_content_1b;
    /**
     * The Coeff content 1 c.
     */
    public final double coeff_content_1c;
    /**
     * The Coeff content 1 d.
     */
    public final double coeff_content_1d;
    /**
     * The Coeff content 1 e.
     */
    public final double coeff_content_1e;
  
    /**
     * Instantiates a new Content coefficients.
     *
     * @param coeff_content_0  the coeff content 0
     * @param coeff_content_1a the coeff content 1 a
     * @param coeff_content_1b the coeff content 1 b
     * @param coeff_content_1c the coeff content 1 c
     * @param coeff_content_1d the coeff content 1 d
     * @param coeff_content_1e the coeff content 1 e
     */
    public ContentCoefficients(final double coeff_content_0, final double coeff_content_1a, final double coeff_content_1b, final double coeff_content_1c, final double coeff_content_1d, final double coeff_content_1e) {
      this.coeff_content_0 = coeff_content_0;
      this.coeff_content_1a = coeff_content_1a;
      this.coeff_content_1b = coeff_content_1b;
      this.coeff_content_1c = coeff_content_1c;
      this.coeff_content_1d = coeff_content_1d;
      this.coeff_content_1e = coeff_content_1e;
    }
    
  }
  
  /**
   * The type Style coefficients.
   */
  public static class StyleCoefficients {
    /**
     * The Coeff style mean 0.
     */
    public final double coeff_style_mean_0;
    /**
     * The Coeff style cov 0.
     */
    public final double coeff_style_cov_0;
    /**
     * The Coeff style mean 1 a.
     */
    public final double coeff_style_mean_1a;
    /**
     * The Coeff style cov 1 a.
     */
    public final double coeff_style_cov_1a;
    /**
     * The Coeff style mean 1 b.
     */
    public final double coeff_style_mean_1b;
    /**
     * The Coeff style cov 1 b.
     */
    public final double coeff_style_cov_1b;
    /**
     * The Coeff style mean 1 c.
     */
    public final double coeff_style_mean_1c;
    /**
     * The Coeff style cov 1 c.
     */
    public final double coeff_style_cov_1c;
    /**
     * The Coeff style mean 1 d.
     */
    public final double coeff_style_mean_1d;
    /**
     * The Coeff style cov 1 d.
     */
    public final double coeff_style_cov_1d;
    /**
     * The Coeff style mean 1 e.
     */
    public final double coeff_style_mean_1e;
    /**
     * The Coeff style cov 1 e.
     */
    public final double coeff_style_cov_1e;
    /**
     * The Dynamic center.
     */
    public final boolean dynamic_center;
  
    /**
     * Instantiates a new Style coefficients.
     *
     * @param coeff_style_mean_0  the coeff style mean 0
     * @param coeff_style_cov_0   the coeff style cov 0
     * @param coeff_style_mean_1a the coeff style mean 1 a
     * @param coeff_style_cov_1a  the coeff style cov 1 a
     * @param coeff_style_mean_1b the coeff style mean 1 b
     * @param coeff_style_cov_1b  the coeff style cov 1 b
     * @param coeff_style_mean_1c the coeff style mean 1 c
     * @param coeff_style_cov_1c  the coeff style cov 1 c
     * @param coeff_style_mean_1d the coeff style mean 1 d
     * @param coeff_style_cov_1d  the coeff style cov 1 d
     * @param coeff_style_mean_1e the coeff style mean 1 e
     * @param coeff_style_cov_1e  the coeff style cov 1 e
     * @param dynamic_center      the dynamic center
     */
    public StyleCoefficients(final double coeff_style_mean_0, final double coeff_style_cov_0, final double coeff_style_mean_1a, final double coeff_style_cov_1a, final double coeff_style_mean_1b, final double coeff_style_cov_1b, final double coeff_style_mean_1c, final double coeff_style_cov_1c, final double coeff_style_mean_1d, final double coeff_style_cov_1d, final double coeff_style_mean_1e, final double coeff_style_cov_1e, final boolean dynamic_center) {
      this.coeff_style_mean_0 = coeff_style_mean_0;
      this.coeff_style_cov_0 = coeff_style_cov_0;
      this.coeff_style_mean_1a = coeff_style_mean_1a;
      this.coeff_style_cov_1a = coeff_style_cov_1a;
      this.coeff_style_mean_1b = coeff_style_mean_1b;
      this.coeff_style_cov_1b = coeff_style_cov_1b;
      this.coeff_style_mean_1c = coeff_style_mean_1c;
      this.coeff_style_cov_1c = coeff_style_cov_1c;
      this.coeff_style_mean_1d = coeff_style_mean_1d;
      this.coeff_style_cov_1d = coeff_style_cov_1d;
      this.coeff_style_mean_1e = coeff_style_mean_1e;
      this.coeff_style_cov_1e = coeff_style_cov_1e;
      this.dynamic_center = dynamic_center;
    }
    
  }
  
  /**
   * The type Style setup.
   */
  public static class StyleSetup {
    /**
     * The Precision.
     */
    public final Precision precision;
    /**
     * The Content image.
     */
    public final transient BufferedImage contentImage;
    /**
     * The Style image.
     */
    public final transient Map<String, BufferedImage> styleImages;
    /**
     * The Styles.
     */
    public final Map<String, StyleCoefficients> styles;
    /**
     * The Content.
     */
    public final ContentCoefficients content;
    /**
     * The Power.
     */
    public double power;
  
  
    /**
     * Instantiates a new Style setup.
     *
     * @param precision           the precision
     * @param contentImage        the content image
     * @param contentCoefficients the content coefficients
     * @param styleImages         the style image
     * @param styles              the styles
     * @param power               the power
     */
    public StyleSetup(final Precision precision, final BufferedImage contentImage, ContentCoefficients contentCoefficients, final Map<String, BufferedImage> styleImages, final Map<String, StyleCoefficients> styles, final double power) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.styleImages = styleImages;
      this.styles = styles;
      this.content = contentCoefficients;
      this.power = power;
    }
    
  }
  
  /**
   * The type Content target.
   */
  public static class ContentTarget {
    /**
     * The Target content 0.
     */
    public Tensor target_content_0;
    /**
     * The Target content 1 a.
     */
    public Tensor target_content_1a;
    /**
     * The Target content 1 b.
     */
    public Tensor target_content_1b;
    /**
     * The Target content 1 c.
     */
    public Tensor target_content_1c;
    /**
     * The Target content 1 d.
     */
    public Tensor target_content_1d;
    /**
     * The Target content 1 e.
     */
    public Tensor target_content_1e;
  }
  
  /**
   * The type Style target.
   */
  public class StyleTarget {
    /**
     * The Target style cov 0.
     */
    public Tensor target_style_cov_0;
    /**
     * The Target style cov 1 a.
     */
    public Tensor target_style_cov_1a;
    /**
     * The Target style cov 1 b.
     */
    public Tensor target_style_cov_1b;
    /**
     * The Target style cov 1 c.
     */
    public Tensor target_style_cov_1c;
    /**
     * The Target style cov 1 d.
     */
    public Tensor target_style_cov_1d;
    /**
     * The Target style cov 1 e.
     */
    public Tensor target_style_cov_1e;
    /**
     * The Target style mean 0.
     */
    public Tensor target_style_mean_0;
    /**
     * The Target style mean 1 a.
     */
    public Tensor target_style_mean_1a;
    /**
     * The Target style mean 1 b.
     */
    public Tensor target_style_mean_1b;
    /**
     * The Target style mean 1 c.
     */
    public Tensor target_style_mean_1c;
    /**
     * The Target style mean 1 d.
     */
    public Tensor target_style_mean_1d;
    /**
     * The Target style mean 1 e.
     */
    public Tensor target_style_mean_1e;
  
    /**
     * The Target style pca 0.
     */
    public Tensor target_style_pca_0;
    /**
     * The Target style pca cov 0.
     */
    public Tensor target_style_pca_cov_0;
    /**
     * The Target style pca 1 a.
     */
    public Tensor target_style_pca_1a;
    /**
     * The Target style pca cov 1 a.
     */
    public Tensor target_style_pca_cov_1a;
    /**
     * The Target style pca 1 b.
     */
    public Tensor target_style_pca_1b;
    /**
     * The Target style pca cov 1 b.
     */
    public Tensor target_style_pca_cov_1b;
    /**
     * The Target style pca 1 c.
     */
    public Tensor target_style_pca_1c;
    /**
     * The Target style pca cov 1 c.
     */
    public Tensor target_style_pca_cov_1c;
    /**
     * The Target style pca 1 d.
     */
    public Tensor target_style_pca_1d;
    /**
     * The Target style pca cov 1 d.
     */
    public Tensor target_style_pca_cov_1d;
    /**
     * The Target style pca 1 e.
     */
    public Tensor target_style_pca_1e;
    /**
     * The Target style pca cov 1 e.
     */
    public Tensor target_style_pca_cov_1e;
  }
  
  /**
   * Gets style components.
   *
   * @param node              the node
   * @param dynamic_center    the dynamic center
   * @param coeff_style_mean  the coeff style mean
   * @param target_style_mean the target style mean
   * @param coeff_style_cov   the coeff style cov
   * @param target_style_cov  the target style cov
   * @param target_style_pca  the target style pca
   * @return the style components
   */
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(final DAGNode node, final boolean dynamic_center, final double coeff_style_mean, final Tensor target_style_mean, final double coeff_style_cov, final Tensor target_style_cov, final Tensor target_style_pca) {
    ArrayList<Tuple2<Double, DAGNode>> list = new ArrayList<>();
    final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
    if (coeff_style_cov != 0 || coeff_style_mean != 0) {
      DAGNode negTarget = network.wrap(new ValueLayer(target_style_mean.scale(-1)), new DAGNode[]{});
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (coeff_style_cov != 0) {
        InnerNode recentered;
        if (dynamic_center) {
          recentered = network.wrap(new GateBiasLayer(), node, negAvg);
        }
        else {
          recentered = network.wrap(new GateBiasLayer(), node, negTarget);
        }
        int[] pcaDim = target_style_pca.getDimensions();
        assert 0 < pcaDim[2] : Arrays.toString(pcaDim);
        int inputBands = target_style_mean.getDimensions()[2];
        assert 0 < inputBands : Arrays.toString(target_style_mean.getDimensions());
        int outputBands = pcaDim[2] / inputBands;
        assert 0 < outputBands : Arrays.toString(pcaDim) + " / " + inputBands;
        list.add(new Tuple2<>(coeff_style_cov, network.wrap(new MeanSqLossLayer(),
          network.wrap(new ValueLayer(target_style_cov), new DAGNode[]{}),
          network.wrap(new GramianLayer(),
            network.wrap(new ConvolutionLayer(pcaDim[0], pcaDim[1], inputBands, outputBands).set(target_style_pca),
              recentered)))
        ));
      }
      if (coeff_style_mean != 0) {
        list.add(new Tuple2<>(coeff_style_mean,
          network.wrap(new MeanSqLossLayer(), negAvg, negTarget)
        ));
      }
    }
    return list;
  }
  
  /**
   * Gets content components.
   *
   * @param node           the node
   * @param coeff_content  the coeff content
   * @param target_content the target content
   * @return the content components
   */
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(final DAGNode node, final double coeff_content, final Tensor target_content) {
    ArrayList<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
    if (coeff_content != 0) {
      functions.add(new Tuple2<>(coeff_content, network.wrap(new MeanSqLossLayer(),
        node, network.wrap(new ValueLayer(target_content), new DAGNode[]{}))));
    }
    return functions;
  }
  
  public class NeuralSetup {
  
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The Style parameters.
     */
    public final StyleSetup style;
    /**
     * The Content target.
     */
    ContentTarget contentTarget = new ContentTarget();
    /**
     * The Style targets.
     */
    Map<String, StyleTarget> styleTargets = new HashMap<>();
    
    /**
     * Instantiates a new Neural setup.
     *
     * @param log   the log
     * @param style the style parameters
     */
    public NeuralSetup(final NotebookOutput log, final StyleSetup style) {
      this.log = log;
      this.style = style;
    }
  
    /**
     * Init neural setup.
     *
     * @return the neural setup
     */
    public NeuralSetup init() {
      final PipelineNetwork content_0 = texture_0(log);
      final PipelineNetwork content_1a = texture_1a(log);
      final PipelineNetwork content_1b = texture_1b(log);
      final PipelineNetwork content_1c = texture_1c(log);
      final PipelineNetwork content_1d = texture_1d(log);
      final PipelineNetwork content_1e = texture_1e(log);
  
      List<String> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
  
      Tensor contentInput = Tensor.fromRGB(style.contentImage);
      List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
      IntStream.range(0, keyList.size()).forEach(i -> {
        styleTargets.put(keyList.get(i), new StyleTarget());
      });
      contentTarget = new ContentTarget();
  
      System.gc();
      contentTarget.target_content_0 = content_0.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_0=" + contentTarget.target_content_0.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_0 = avg((PipelineNetwork) content_0.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_0=" + styleTarget.target_style_mean_0.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_0 = gram((PipelineNetwork) content_0.copy(), styleTarget.target_style_mean_0).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_0=" + styleTarget.target_style_cov_0.prettyPrint());
        styleTarget.target_style_pca_0 = pca(styleTarget.target_style_cov_0, style.power);
        logger.info("target_style_pca_0=" + styleTarget.target_style_pca_0.prettyPrint());
        styleTarget.target_style_pca_cov_0 = gram((PipelineNetwork) content_0.copy(), styleTarget.target_style_mean_0, styleTarget.target_style_pca_0).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_0=" + styleTarget.target_style_pca_cov_0.prettyPrint());
      }
  
      System.gc();
      contentTarget.target_content_1a = content_1a.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1a=" + contentTarget.target_content_1a.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_1a = avg((PipelineNetwork) content_1a.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1a=" + styleTarget.target_style_mean_1a.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_1a = gram((PipelineNetwork) content_1a.copy(), styleTarget.target_style_mean_1a).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1a=" + styleTarget.target_style_cov_1a.prettyPrint());
        styleTarget.target_style_pca_1a = pca(styleTarget.target_style_cov_1a, style.power);
        logger.info("target_style_pca_1a=" + styleTarget.target_style_pca_1a.prettyPrint());
        styleTarget.target_style_pca_cov_1a = gram((PipelineNetwork) content_1a.copy(), styleTarget.target_style_mean_1a, styleTarget.target_style_pca_1a).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_1a=" + styleTarget.target_style_pca_cov_1a.prettyPrint());
      }
  
      System.gc();
      contentTarget.target_content_1b = content_1b.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1b=" + contentTarget.target_content_1b.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_1b = avg((PipelineNetwork) content_1b.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1b=" + styleTarget.target_style_mean_1b.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_1b = gram((PipelineNetwork) content_1b.copy(), styleTarget.target_style_mean_1b).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1b=" + styleTarget.target_style_cov_1b.prettyPrint());
        styleTarget.target_style_pca_1b = pca(styleTarget.target_style_cov_1b, style.power);
        logger.info("target_style_pca_1b=" + styleTarget.target_style_pca_1b.prettyPrint());
        styleTarget.target_style_pca_cov_1b = gram((PipelineNetwork) content_1b.copy(), styleTarget.target_style_mean_1b, styleTarget.target_style_pca_1b).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_1b=" + styleTarget.target_style_pca_cov_1b.prettyPrint());
      }
  
      System.gc();
      contentTarget.target_content_1c = content_1c.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1c=" + contentTarget.target_content_1c.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_1c = avg((PipelineNetwork) content_1c.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1c=" + styleTarget.target_style_mean_1c.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_1c = gram((PipelineNetwork) content_1c.copy(), styleTarget.target_style_mean_1c).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1c=" + styleTarget.target_style_cov_1c.prettyPrint());
        styleTarget.target_style_pca_1c = pca(styleTarget.target_style_cov_1c, style.power);
        logger.info("target_style_pca_1c=" + styleTarget.target_style_pca_1c.prettyPrint());
        styleTarget.target_style_pca_cov_1c = gram((PipelineNetwork) content_1c.copy(), styleTarget.target_style_mean_1c, styleTarget.target_style_pca_1c).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_1c=" + styleTarget.target_style_pca_cov_1c.prettyPrint());
      }
  
      System.gc();
      contentTarget.target_content_1d = content_1d.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1d=" + contentTarget.target_content_1d.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_1d = avg((PipelineNetwork) content_1d.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1d=" + styleTarget.target_style_mean_1d.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_1d = gram((PipelineNetwork) content_1d.copy(), styleTarget.target_style_mean_1d).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1d=" + styleTarget.target_style_cov_1d.prettyPrint());
        styleTarget.target_style_pca_1d = pca(styleTarget.target_style_cov_1d, style.power);
        logger.info("target_style_pca_1d=" + styleTarget.target_style_pca_1d.prettyPrint());
        styleTarget.target_style_pca_cov_1d = gram((PipelineNetwork) content_1d.copy(), styleTarget.target_style_mean_1d, styleTarget.target_style_pca_1d).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_1d=" + styleTarget.target_style_pca_cov_1d.prettyPrint());
      }
  
      System.gc();
      contentTarget.target_content_1e = content_1e.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1e=" + contentTarget.target_content_1e.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.target_style_mean_1e = avg((PipelineNetwork) content_1e.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1e=" + styleTarget.target_style_mean_1e.prettyPrint());
        System.gc();
        styleTarget.target_style_cov_1e = gram((PipelineNetwork) content_1e.copy(), styleTarget.target_style_mean_1e).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1e=" + styleTarget.target_style_cov_1e.prettyPrint());
        styleTarget.target_style_pca_1e = pca(styleTarget.target_style_cov_1e, style.power);
        logger.info("target_style_pca_1e=" + styleTarget.target_style_pca_1e.prettyPrint());
        styleTarget.target_style_pca_cov_1e = gram((PipelineNetwork) content_1e.copy(), styleTarget.target_style_mean_1e, styleTarget.target_style_pca_1e).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_pca_cov_1e=" + styleTarget.target_style_pca_cov_1e.prettyPrint());
      }
  
      return this;
    }
  
    /**
     * Fitness function pipeline network.
     *
     * @param log the log
     * @return the pipeline network
     */
    @Nonnull
    public PipelineNetwork fitnessNetwork(final NotebookOutput log) {
      final PipelineNetwork[] layerBuffer = new PipelineNetwork[1];
      final DAGNode[] nodes = new DAGNode[6];
      try {
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase0() {
            super.phase0();
            nodes[0] = pipeline.getHead();
          }
      
          @Override
          protected void phase1a() {
            super.phase1a();
            nodes[1] = pipeline.getHead();
          }
      
          @Override
          protected void phase1b() {
            super.phase1b();
            nodes[2] = pipeline.getHead();
          }
      
          @Override
          protected void phase1c() {
            super.phase1c();
            nodes[3] = pipeline.getHead();
          }
      
          @Override
          protected void phase1d() {
            super.phase1d();
            nodes[4] = pipeline.getHead();
          }
      
          @Override
          protected void phase1e() {
            super.phase1e();
            nodes[5] = pipeline.getHead();
            layerBuffer[0] = (PipelineNetwork) pipeline.freeze();
            throw new RuntimeException("Abort Network Construction");
          }
        }.getNetwork();
      } catch (@Nonnull final RuntimeException e1) {
      } catch (Throwable e11) {
        throw new RuntimeException(e11);
      }
      List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
  
      {
        ContentCoefficients c = this.style.content;
        functions.addAll(getContentComponents(nodes[0], c.coeff_content_0, contentTarget.target_content_0));
        functions.addAll(getContentComponents(nodes[1], c.coeff_content_1a, contentTarget.target_content_1a));
        functions.addAll(getContentComponents(nodes[2], c.coeff_content_1b, contentTarget.target_content_1b));
        functions.addAll(getContentComponents(nodes[3], c.coeff_content_1c, contentTarget.target_content_1c));
        functions.addAll(getContentComponents(nodes[4], c.coeff_content_1d, contentTarget.target_content_1d));
        functions.addAll(getContentComponents(nodes[5], c.coeff_content_1e, contentTarget.target_content_1e));
      }
  
      for (final String key : this.style.styles.keySet()) {
        StyleTarget t = this.styleTargets.get(key);
        StyleCoefficients c = this.style.styles.get(key);
        assert null != c;
        assert null != t;
        functions.addAll(getStyleComponents(nodes[0], c.dynamic_center, c.coeff_style_mean_0, t.target_style_mean_0, c.coeff_style_cov_0, t.target_style_pca_cov_0, t.target_style_pca_0));
        functions.addAll(getStyleComponents(nodes[1], c.dynamic_center, c.coeff_style_mean_1a, t.target_style_mean_1a, c.coeff_style_cov_1a, t.target_style_pca_cov_1a, t.target_style_pca_1a));
        functions.addAll(getStyleComponents(nodes[2], c.dynamic_center, c.coeff_style_mean_1b, t.target_style_mean_1b, c.coeff_style_cov_1b, t.target_style_pca_cov_1b, t.target_style_pca_1b));
        functions.addAll(getStyleComponents(nodes[3], c.dynamic_center, c.coeff_style_mean_1c, t.target_style_mean_1c, c.coeff_style_cov_1c, t.target_style_pca_cov_1c, t.target_style_pca_1c));
        functions.addAll(getStyleComponents(nodes[4], c.dynamic_center, c.coeff_style_mean_1d, t.target_style_mean_1d, c.coeff_style_cov_1d, t.target_style_pca_cov_1d, t.target_style_pca_1d));
        functions.addAll(getStyleComponents(nodes[5], c.dynamic_center, c.coeff_style_mean_1e, t.target_style_mean_1e, c.coeff_style_cov_1e, t.target_style_pca_cov_1e, t.target_style_pca_1e));
      }
  
      PipelineNetwork network1 = layerBuffer[0];
      functions.stream().filter(x -> x._1 != 0)
        .reduce((a, b) -> new Tuple2<>(1.0, network1.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2))).get();
  
      PipelineNetwork network = withClamp(network1);
      setPrecision(network, this.style.precision);
      return network;
    }
    
  }

}
