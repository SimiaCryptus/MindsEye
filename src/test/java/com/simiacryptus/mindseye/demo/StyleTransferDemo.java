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

package com.simiacryptus.mindseye.demo;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.GateBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Image classifier apply base.
 */
public class StyleTransferDemo extends ArtistryDemo {
  
  
  /**
   * The Texture netork.
   */
  int imageSize = 600;
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run);
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init(log);
    Precision precision = Precision.Float;
    imageSize = 300;
//    String content = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170924_145214.jpg";
    String content = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
//    String style = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  
    log.h1("Input");
    BufferedImage canvasImage = load(content, imageSize);
    canvasImage = randomize(canvasImage);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    BufferedImage contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    List<BufferedImage> styleImages = load(Arrays.asList(
      "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\1280px-Van_Gogh_-_Kauernder_Junge_mit_Sichel.jpg",
      "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg"
    ), imageSize);
    ContentCoefficients contentCoefficients = new ContentCoefficients(
      1,
      1e2,
      1e3,
      1e3,
      0,
      0);
    List<StyleCoefficients> styleCoefficients = styleImages.stream().map(img -> new StyleCoefficients(
      1, 1,
      1, 1,
      1, 1,
      1, 1,
      0, 0,
      0, 0, true)).collect(Collectors.toList());
    int trainingMinutes = 1;
  
    canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styleCoefficients), trainingMinutes);
    
    for (int i = 0; i < 3; i++) {
      imageSize = imageSize * 2;
      canvasImage = TestUtil.resize(canvasImage, imageSize, true);
      contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
      styleImages = load(Arrays.asList(
        "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\1280px-Van_Gogh_-_Kauernder_Junge_mit_Sichel.jpg",
        "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg"
      ), imageSize);
      canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styleCoefficients), trainingMinutes);
    }
  
    imageSize = imageSize * 2;
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    styleImages = load(Arrays.asList(
      "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\1280px-Van_Gogh_-_Kauernder_Junge_mit_Sichel.jpg",
      "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg"
    ), imageSize);
    canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styleCoefficients), trainingMinutes);
    
    log.setFrontMatterProperty("status", "OK");
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
   * Style transfer buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup styleParameters, final int trainingMinutes) {
    NeuralSetup neuralSetup = new NeuralSetup(log, styleParameters).init();
    PipelineNetwork network = neuralSetup.fitnessNetwork(log);
    BufferedImage result = train(log, canvasImage, network, styleParameters.precision, trainingMinutes);
    try {
      log.p(log.image(result, "result"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
  
  /**
   * Train buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param network         the network
   * @param precision       the precision
   * @param trainingMinutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage train(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final PipelineNetwork network, final Precision precision, final int trainingMinutes) {
    log.h1("Output");
    Tensor canvas = Tensor.fromRGB(canvasImage);
    TestUtil.monitorImage(canvas, false);
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
        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        .setTimeout(trainingMinutes, TimeUnit.MINUTES)
        .runAndFree();
      return TestUtil.plot(history);
    });
    return canvas.toImage();
  }
  
  @Nonnull
  public List<BufferedImage> load(final List<String> style, final int width, final int height) {
    return style.stream().map(x -> load(x, width, height)).collect(Collectors.toList());
  }
  
  @Nonnull
  public List<BufferedImage> load(final List<String> style, final int imageSize) {
    return style.stream().map(x -> load(x, imageSize)).collect(Collectors.toList());
  }
  
  /**
   * Load buffered image.
   *
   * @param style     the style
   * @param imageSize the image size
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage load(final String style, final int imageSize) {
    BufferedImage image1;
    try {
      image1 = ImageIO.read(new File(style));
      image1 = TestUtil.resize(image1, imageSize, true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return image1;
  }
  
  /**
   * Load buffered image.
   *
   * @param style  the style
   * @param width  the width
   * @param height the height
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage load(final String style, final int width, final int height) {
    BufferedImage image1;
    try {
      image1 = ImageIO.read(new File(style));
      image1 = TestUtil.resize(image1, width, height);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return image1;
  }
  
  /**
   * Texture 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork texture_0(@Nonnull final NotebookOutput log) {
    final PipelineNetwork[] layers = new PipelineNetwork[1];
    log.code(() -> {
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
    });
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
    log.code(() -> {
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
    });
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
    log.code(() -> {
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
    });
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
    log.code(() -> {
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
    });
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
    log.code(() -> {
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
    });
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
    log.code(() -> {
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
    });
    return layers[0];
  }
  
  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork gram(final PipelineNetwork network) {
    network.wrap(new GramianLayer());
    return network;
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
   * Sets precision.
   *
   * @param network   the network
   * @param precision the precision
   */
  public void setPrecision(final DAGNetwork network, final Precision precision) {
    network.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
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
    return ReportType.Demos;
  }
  
  public static class ContentCoefficients {
    public final double coeff_content_0;
    public final double coeff_content_1a;
    public final double coeff_content_1b;
    public final double coeff_content_1c;
    public final double coeff_content_1d;
    public final double coeff_content_1e;
    
    public ContentCoefficients(final double coeff_content_0, final double coeff_content_1a, final double coeff_content_1b, final double coeff_content_1c, final double coeff_content_1d, final double coeff_content_1e) {
      this.coeff_content_0 = coeff_content_0;
      this.coeff_content_1a = coeff_content_1a;
      this.coeff_content_1b = coeff_content_1b;
      this.coeff_content_1c = coeff_content_1c;
      this.coeff_content_1d = coeff_content_1d;
      this.coeff_content_1e = coeff_content_1e;
    }
    
  }
  
  public static class StyleCoefficients {
    public final double coeff_style_mean_0;
    public final double coeff_style_cov_0;
    public final double coeff_style_mean_1a;
    public final double coeff_style_cov_1a;
    public final double coeff_style_mean_1b;
    public final double coeff_style_cov_1b;
    public final double coeff_style_mean_1c;
    public final double coeff_style_cov_1c;
    public final double coeff_style_mean_1d;
    public final double coeff_style_cov_1d;
    public final double coeff_style_mean_1e;
    public final double coeff_style_cov_1e;
    public final boolean dynamic_center;
    
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
    public final BufferedImage contentImage;
    /**
     * The Style image.
     */
    public final List<BufferedImage> styleImages;
    public final List<StyleCoefficients> styles;
    public final ContentCoefficients content;
  
  
    /**
     * Instantiates a new Style setup.
     *
     * @param precision    the precision
     * @param contentImage the content image
     * @param styleImages   the style image
     */
    public StyleSetup(final Precision precision, final BufferedImage contentImage, ContentCoefficients contentCoefficients, final List<BufferedImage> styleImages, final List<StyleCoefficients> styles) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.styleImages = styleImages;
      this.styles = styles;
      this.content = contentCoefficients;
    }
    
  }
  
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
  }
  
  private class NeuralSetup {
    
    /**
     * The Log.
     */
    public final NotebookOutput log;
    /**
     * The Style parameters.
     */
    public final StyleSetup style;
    ContentTarget contentTarget = new ContentTarget();
    List<StyleTarget> styleTargets = new ArrayList<>();
    
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
      
      Tensor contentInput = Tensor.fromRGB(style.contentImage);
      try {
        log.p(log.image(contentInput.toImage(), "content"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
  
      List<Tensor> styleInputs = style.styleImages.stream().map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
      styleInputs.forEach(styleInput -> {
        try {
          log.p(log.image(styleInput.toImage(), "style"));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
        styleTargets.add(new StyleTarget());
      });
      contentTarget = new ContentTarget();
      
      contentTarget.target_content_0 = content_0.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_0=" + contentTarget.target_content_0.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_0 = avg((PipelineNetwork) content_0.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_0=" + styleTarget.target_style_mean_0.prettyPrint());
        styleTarget.target_style_cov_0 = gram((PipelineNetwork) content_0.copy(), styleTarget.target_style_mean_0).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_0=" + styleTarget.target_style_cov_0.prettyPrint());
      }
      
      contentTarget.target_content_1a = content_1a.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1a=" + contentTarget.target_content_1a.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_1a = avg((PipelineNetwork) content_1a.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1a=" + styleTarget.target_style_mean_1a.prettyPrint());
        styleTarget.target_style_cov_1a = gram((PipelineNetwork) content_1a.copy(), styleTarget.target_style_mean_1a).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1a=" + styleTarget.target_style_cov_1a.prettyPrint());
      }
      
      contentTarget.target_content_1b = content_1b.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1b=" + contentTarget.target_content_1b.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_1b = avg((PipelineNetwork) content_1b.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1b=" + styleTarget.target_style_mean_1b.prettyPrint());
        styleTarget.target_style_cov_1b = gram((PipelineNetwork) content_1b.copy(), styleTarget.target_style_mean_1b).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1b=" + styleTarget.target_style_cov_1b.prettyPrint());
      }
      
      contentTarget.target_content_1c = content_1c.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1c=" + contentTarget.target_content_1c.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_1c = avg((PipelineNetwork) content_1c.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1c=" + styleTarget.target_style_mean_1c.prettyPrint());
        styleTarget.target_style_cov_1c = gram((PipelineNetwork) content_1c.copy(), styleTarget.target_style_mean_1c).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1c=" + styleTarget.target_style_cov_1c.prettyPrint());
      }
      
      contentTarget.target_content_1d = content_1d.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1d=" + contentTarget.target_content_1d.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_1d = avg((PipelineNetwork) content_1d.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1d=" + styleTarget.target_style_mean_1d.prettyPrint());
        styleTarget.target_style_cov_1d = gram((PipelineNetwork) content_1d.copy(), styleTarget.target_style_mean_1d).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1d=" + styleTarget.target_style_cov_1d.prettyPrint());
      }
      
      contentTarget.target_content_1e = content_1e.eval(contentInput).getDataAndFree().getAndFree(0);
      logger.info("target_content_1e=" + contentTarget.target_content_1e.prettyPrint());
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget styleTarget = styleTargets.get(i);
        styleTarget.target_style_mean_1e = avg((PipelineNetwork) content_1e.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_mean_1e=" + styleTarget.target_style_mean_1e.prettyPrint());
        styleTarget.target_style_cov_1e = gram((PipelineNetwork) content_1e.copy(), styleTarget.target_style_mean_1e).eval(styleInput).getDataAndFree().getAndFree(0);
        logger.info("target_style_cov_1e=" + styleTarget.target_style_cov_1e.prettyPrint());
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
      log.code(() -> {
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
      });
      PipelineNetwork network = layerBuffer[0];
      List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
  
      {
        ContentCoefficients c = this.style.content;
        addContentComponents(network, functions, nodes[0],
          c.coeff_content_0, contentTarget.target_content_0);
        addContentComponents(network, functions, nodes[1],
          c.coeff_content_1a, contentTarget.target_content_1a);
        addContentComponents(network, functions, nodes[2],
          c.coeff_content_1b, contentTarget.target_content_1b);
        addContentComponents(network, functions, nodes[3],
          c.coeff_content_1c, contentTarget.target_content_1c);
        addContentComponents(network, functions, nodes[4],
          c.coeff_content_1d, contentTarget.target_content_1d);
        addContentComponents(network, functions, nodes[5],
          c.coeff_content_1e, contentTarget.target_content_1e);
      }
  
      for (int i = 0; i < this.style.styles.size(); i++) {
        StyleTarget styleTarget = this.styleTargets.get(i);
        StyleCoefficients c = this.style.styles.get(i);
        addStyleComponents(network, functions, nodes[0], c.dynamic_center,
          c.coeff_style_mean_0, styleTarget.target_style_mean_0,
          c.coeff_style_cov_0, styleTarget.target_style_cov_0);
        addStyleComponents(network, functions, nodes[1], c.dynamic_center,
          c.coeff_style_mean_1a, styleTarget.target_style_mean_1a,
          c.coeff_style_cov_1a, styleTarget.target_style_cov_1a);
        addStyleComponents(network, functions, nodes[2], c.dynamic_center,
          c.coeff_style_mean_1b, styleTarget.target_style_mean_1b,
          c.coeff_style_cov_1b, styleTarget.target_style_cov_1b);
        addStyleComponents(network, functions, nodes[3], c.dynamic_center,
          c.coeff_style_mean_1c, styleTarget.target_style_mean_1c,
          c.coeff_style_cov_1c, styleTarget.target_style_cov_1c);
        addStyleComponents(network, functions, nodes[4], c.dynamic_center,
          c.coeff_style_mean_1d, styleTarget.target_style_mean_1d,
          c.coeff_style_cov_1d, styleTarget.target_style_cov_1d);
        addStyleComponents(network, functions, nodes[5], c.dynamic_center,
          c.coeff_style_mean_1e, styleTarget.target_style_mean_1e,
          c.coeff_style_cov_1e, styleTarget.target_style_cov_1e);
      }
      
      functions.stream().filter(x -> x._1 != 0)
        .reduce((a, b) -> new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2))).get();
      setPrecision(network, this.style.precision);
      return network;
    }
    
    public void addStyleComponents(final PipelineNetwork network, final List<Tuple2<Double, DAGNode>> functions, final DAGNode node, final boolean dynamic_center, final double coeff_style_mean, final Tensor target_style_mean, final double coeff_style_cov, final Tensor target_style_cov) {
      if (coeff_style_cov != 0 || coeff_style_mean != 0) {
        DAGNode negTarget = network.constValue(target_style_mean.scale(-1));
        InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1).setMode(PoolingLayer.PoolingMode.Avg), node);
        if (coeff_style_cov != 0) {
          InnerNode recentered;
          if (dynamic_center) {
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
          }
          else {
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
          }
          functions.add(new Tuple2<>(coeff_style_cov, network.wrap(new MeanSqLossLayer(),
            network.constValue(target_style_cov),
            network.wrap(new GramianLayer(), recentered))
          ));
        }
        if (coeff_style_mean != 0) {
          functions.add(new Tuple2<>(coeff_style_mean,
            network.wrap(new MeanSqLossLayer(), negAvg, negTarget)
          ));
        }
      }
    }
    
    public void addContentComponents(final PipelineNetwork network, final List<Tuple2<Double, DAGNode>> functions, final DAGNode node, final double coeff_content, final Tensor target_content) {
      if (coeff_content != 0) {
        functions.add(new Tuple2<>(coeff_content, network.wrap(new MeanSqLossLayer(),
          node, network.constValue(target_content))));
      }
    }
  }
}
