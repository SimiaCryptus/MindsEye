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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
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
    String content = "H:\\SimiaCryptus\\Artistry\\monkeydog.jpg";
    String style = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  
    log.h1("Input");
    final PipelineNetwork texture_1d = texture_1d(log);
    setPrecision(texture_1d, precision);
    final PipelineNetwork style_1d = style_1d(log);
    setPrecision(style_1d, precision);
    Tensor contentInput = loadImage(content);
    try {
      log.p(log.image(contentInput.toImage(), "content"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    Tensor styleInput = loadImage(style);
    try {
      log.p(log.image(styleInput.toImage(), "style"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    Tensor target_texture_1d = texture_1d.eval(contentInput).getDataAndFree().getAndFree(0);
    Tensor target_style_1d = style_1d.eval(styleInput).getDataAndFree().getAndFree(0);
    PipelineNetwork loss_1d = loss_1d(log, target_texture_1d, target_style_1d);
    setPrecision(loss_1d, precision);
  
    log.h1("Output");
    Tensor canvas = contentInput.map(x -> FastRandom.INSTANCE.random());
    TestUtil.monitorImage(canvas, false);
    @Nonnull Trainable trainable = new ArrayTrainable(loss_1d, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    TestUtil.instrumentPerformance(log, loss_1d);
    addLayersHandler(loss_1d, server);
  
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(180, TimeUnit.MINUTES)
        .runAndFree();
      return TestUtil.plot(history);
    });
  
    try {
      log.p(log.image(canvas.toImage(), "result"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Load image tensor.
   *
   * @param style the style
   * @return the tensor
   */
  @Nonnull
  public Tensor loadImage(final String style) {
    Tensor contentInput;
    try {
      BufferedImage image1 = ImageIO.read(new File(style));
      image1 = TestUtil.resize(image1, imageSize, true);
      contentInput = Tensor.fromRGB(image1);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return contentInput;
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
   * Style 1 d pipeline network.
   *
   * @param log the log
   * @return the pipeline network
   */
  public PipelineNetwork style_1d(@Nonnull final NotebookOutput log) {
    final Layer[] layers = new Layer[1];
    log.code(() -> {
      try {
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase1d() {
            layers[0] = pipeline.freeze();
            throw new RuntimeException("Abort Network Construction");
          }
        }.getNetwork();
      } catch (@Nonnull final RuntimeException e) {
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
    });
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(new GramianLayer(),
      network.wrap(layers[0],
        network.getInput(0)));
    return network;
  }
  
  /**
   * Loss 1 d pipeline network.
   *
   * @param log        the log
   * @param texture_1d the texture 1 d
   * @param style_1d   the style 1 d
   * @return the pipeline network
   */
  public PipelineNetwork loss_1d(@Nonnull final NotebookOutput log, Tensor texture_1d, Tensor style_1d) {
    final PipelineNetwork[] pipelineNetwork = new PipelineNetwork[1];
    log.code(() -> {
      try {
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase1d() {
            pipelineNetwork[0] = (PipelineNetwork) pipeline.freeze();
            throw new RuntimeException("Abort Network Construction");
          }
        }.getNetwork();
      } catch (@Nonnull final RuntimeException e) {
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
    });
    PipelineNetwork network = pipelineNetwork[0];
    DAGNode texture1d = network.getHead();
    network.wrap(new BinarySumLayer(1.0, 1.0),
      network.wrap(new MeanSqLossLayer(), texture1d, network.constValue(texture_1d)),
      network.wrap(new MeanSqLossLayer(), network.wrap(new GramianLayer(), texture1d), network.constValue(style_1d))
    );
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
