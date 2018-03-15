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
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
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
    String content = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170924_145214.jpg";
    String style = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  
    log.h1("Input");
    BufferedImage canvasImage = randomize(load(content, imageSize));
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    BufferedImage contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    BufferedImage styleImage = load(style, imageSize);
    canvasImage = styleTransfer(log, precision, contentImage, styleImage, canvasImage,
      0.0, 1,
      1, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0);
  
    imageSize = imageSize * 2;
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    styleImage = load(style, imageSize);
    canvasImage = styleTransfer(log, precision, contentImage, styleImage, canvasImage,
      0.0, 0.0,
      0.0, 0.0,
      10.0, 1.0,
      10.0, 1.0,
      10.0, 1.0,
      0.0, 0.0);
  
    imageSize = imageSize * 2;
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    styleImage = load(style, imageSize);
    canvasImage = styleTransfer(log, precision, contentImage, styleImage, canvasImage,
      0.0, 0.0,
      0.0, 0.0,
      0.0, 0.0,
      1000.0, 1.0,
      1000.0, 1.0,
      1000.0, 1.0);
  
    imageSize = imageSize * 2;
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    contentImage = load(content, canvasImage.getWidth(), canvasImage.getHeight());
    styleImage = load(style, imageSize);
    canvasImage = styleTransfer(log, precision, contentImage, styleImage, canvasImage, 0.0 * 1e3 * (double) 0, 1.0 * (double) 0, 1e3 * (double) 0, 1.0 * (double) 0, 1e3 * (double) 0, 1.0 * (double) 0, 1e3 * 1.0, 1.0 * 1.0, 1e3 * 1.0, 1.0 * 1.0, 1e3 * 1.0, 1.0 * 1.0);
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  @Nonnull
  public BufferedImage randomize(final BufferedImage contentImage) {
    return Tensor.fromRGB(contentImage).map(x -> FastRandom.INSTANCE.random()).toRgbImage();
  }
  
  @Nonnull
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final Precision precision, final BufferedImage contentImage, final BufferedImage styleImage, final BufferedImage canvasImage, final double coeff_content_0, final double coeff_style_0, final double coeff_content_1a, final double coeff_style_1a, final double coeff_content_1b, final double coeff_style_1b, final double coeff_content_1c, final double coeff_style_1c, final double coeff_content_1d, final double coeff_style_1d, final double coeff_content_1e, final double coeff_style_1e) {
    final PipelineNetwork content_0 = texture_0(log);
    final PipelineNetwork style_0 = gram((PipelineNetwork) content_0.copy());
    final PipelineNetwork content_1a = texture_1a(log);
    final PipelineNetwork style_1a = gram((PipelineNetwork) content_1a.copy());
    final PipelineNetwork content_1b = texture_1b(log);
    final PipelineNetwork style_1b = gram((PipelineNetwork) content_1b.copy());
    final PipelineNetwork content_1c = texture_1c(log);
    final PipelineNetwork style_1c = gram((PipelineNetwork) content_1c.copy());
    final PipelineNetwork content_1d = texture_1d(log);
    final PipelineNetwork style_1d = gram((PipelineNetwork) content_1d.copy());
    final PipelineNetwork content_1e = texture_1e(log);
    final PipelineNetwork style_1e = gram((PipelineNetwork) content_1e.copy());
  
    setPrecision(content_0, precision);
    setPrecision(style_0, precision);
    setPrecision(content_1a, precision);
    setPrecision(style_1a, precision);
    setPrecision(content_1b, precision);
    setPrecision(style_1b, precision);
    setPrecision(content_1c, precision);
    setPrecision(style_1c, precision);
    setPrecision(content_1d, precision);
    setPrecision(style_1d, precision);
    setPrecision(content_1e, precision);
    setPrecision(style_1e, precision);
    
    Tensor contentInput = Tensor.fromRGB(contentImage);
    try {
      log.p(log.image(contentInput.toImage(), "content"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    Tensor styleInput = Tensor.fromRGB(styleImage);
    try {
      log.p(log.image(styleInput.toImage(), "style"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  
    Tensor target_content_0 = content_0.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_0 = style_0.eval(styleInput).getDataAndFree().getAndFree(0);
    Tensor target_content_1a = content_1a.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1a = style_1a.eval(styleInput).getDataAndFree().getAndFree(0);
    Tensor target_content_1b = content_1b.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1b = style_1b.eval(styleInput).getDataAndFree().getAndFree(0);
    Tensor target_content_1c = content_1c.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1c = style_1c.eval(styleInput).getDataAndFree().getAndFree(0);
    Tensor target_content_1d = content_1d.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1d = style_1d.eval(styleInput).getDataAndFree().getAndFree(0);
    Tensor target_content_1e = content_1e.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1e = style_1e.eval(styleInput).getDataAndFree().getAndFree(0);
    
    final PipelineNetwork[] layerBuffer = new PipelineNetwork[1];
    final DAGNode[] nodes = new DAGNode[6];
    log.code(() -> {
      try {
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase0() {
            super.phase0();
            layerBuffer[0] = (PipelineNetwork) pipeline.freeze();
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
            throw new RuntimeException("Abort Network Construction");
          }
        }.getNetwork();
      } catch (@Nonnull final RuntimeException e1) {
      } catch (Throwable e11) {
        throw new RuntimeException(e11);
      }
    });
    List<DAGNode> lossLayers = new ArrayList<>();
    PipelineNetwork network = layerBuffer[0];
  
    if (coeff_content_0 != 0 || coeff_style_0 != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_0, coeff_style_0),
        network.wrap(new MeanSqLossLayer(), nodes[0], network.constValue(target_content_0)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[0]), network.constValue(target_style_0))
      ));
    }
    if (coeff_content_1a != 0 || coeff_style_1a != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_1a, coeff_style_1a),
        network.wrap(new MeanSqLossLayer(), nodes[1], network.constValue(target_content_1a)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[1]), network.constValue(target_style_1a))
      ));
    }
    if (coeff_content_1b != 0 || coeff_style_1b != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_1b, coeff_style_1b),
        network.wrap(new MeanSqLossLayer(), nodes[2], network.constValue(target_content_1b)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[2]), network.constValue(target_style_1b))
      ));
    }
    if (coeff_content_1c != 0 || coeff_style_1c != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_1c, coeff_style_1c),
        network.wrap(new MeanSqLossLayer(), nodes[3], network.constValue(target_content_1c)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[3]), network.constValue(target_style_1c))
      ));
    }
    if (coeff_content_1d != 0 || coeff_style_1d != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_1d, coeff_style_1d),
        network.wrap(new MeanSqLossLayer(), nodes[4], network.constValue(target_content_1d)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[4]), network.constValue(target_style_1d))
      ));
    }
    if (coeff_content_1e != 0 || coeff_style_1e != 0) {
      lossLayers.add(network.wrap(new BinarySumLayer(coeff_content_1e, coeff_style_1e),
        network.wrap(new MeanSqLossLayer(), nodes[5], network.constValue(target_content_1e)),
        network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), nodes[5]), network.constValue(target_style_1e))
      ));
    }
    network.wrap(new SumInputsLayer(), lossLayers.toArray(new DAGNode[]{}));
    setPrecision(network, precision);
    
    log.h1("Output");
    Tensor canvas = Tensor.fromRGB(canvasImage);
    TestUtil.monitorImage(canvas, false);
    @Nonnull Trainable trainable = new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    TestUtil.instrumentPerformance(log, network);
    addLayersHandler(network, server);
    
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(90, TimeUnit.MINUTES)
        .runAndFree();
      return TestUtil.plot(history);
    });
    BufferedImage result = canvas.toImage();
    try {
      log.p(log.image(result, "result"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
  
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
  
  @Nonnull
  public PipelineNetwork gram(final PipelineNetwork network) {
    network.wrap(new GramianLayer());
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
  
}
