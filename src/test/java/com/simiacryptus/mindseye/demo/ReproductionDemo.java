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

import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.hadoop.yarn.webapp.MimeType;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

/**
 * The type Image classifier run base.
 */
public class ReproductionDemo extends ArtistryDemo {
  
  /**
   * The Files.
   */
  public final String[] files;
  
  /**
   * Instantiates a new Reproduction demo.
   *
   * @param files the files
   */
  public ReproductionDemo(final String... files) {
    this.files = null != files && 0 < files.length ? files : new String[]{
      "H:\\SimiaCryptus\\Artistry\\chimps\\winner.jpg",
      "H:\\SimiaCryptus\\Artistry\\chimps\\chip.jpg"
    };
  }
  
  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    ReproductionDemo demo = new ReproductionDemo(args);
    demo.run(demo::run);
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init();
    log.h1("Model");
    Layer textureNetork;
    try {
      textureNetork = getTextureNetwork(log);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  
    final AtomicInteger imageNumber = new AtomicInteger(0);
    Stream.of(
      files
    ).forEach(img -> {
      log.h1("Image " + imageNumber.getAndIncrement());
      try {
        Tensor originalImage = loadImage(img);
        Tensor canvas = getCanvas(log, originalImage);
        TestUtil.monitorImage(canvas, false, 5);
        Tensor texture = getTexture(textureNetork, originalImage);
        paint(log, textureNetork, canvas, texture);
        log.out(log.image(originalImage.toImage(), "Reproduction"));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
  
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Paint.
   *
   * @param log           the log
   * @param textureNetork the texture netork
   * @param canvas        the canvas
   * @param texture       the texture
   */
  public void paint(@Nonnull final NotebookOutput log, final Layer textureNetork, final Tensor canvas, final Tensor texture) {
    @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
    
    @Nonnull PipelineNetwork painterNetwork = new PipelineNetwork(2);
    painterNetwork.wrap(new MeanSqLossLayer(),
      painterNetwork.getInput(0),
      painterNetwork.add(textureNetork,
        painterNetwork.add(clamp,
          painterNetwork.getInput(1))));
    BasicTrainable basicTrainable = new BasicTrainable(painterNetwork).setVerbosity(2);
    basicTrainable.setMask(false, true);
    basicTrainable.setData(Arrays.<Tensor[]>asList(new Tensor[]{texture, canvas}));
    
    TestUtil.instrumentPerformance(log, painterNetwork);
    server.addSyncHandler("layers.json", MimeType.JSON, out -> {
      try {
        JsonUtil.MAPPER.writer().writeValue(out, TestUtil.samplePerformance(painterNetwork));
        out.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }, false);
    
    
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(basicTrainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new QuadraticSearch().setCurrentRate(20).setRelativeTolerance(0.05))
        .setTimeout(8, TimeUnit.HOURS)
        .runAndFree();
      return TestUtil.plot(history);
    });
  }
  
  /**
   * Gets texture.
   *
   * @param textureNetork the texture netork
   * @param image         the image
   * @return the texture
   */
  public Tensor getTexture(final Layer textureNetork, final Tensor image) {
    return textureNetork.eval(image).getDataAndFree().getAndFree(0);
  }
  
  /**
   * Gets canvas.
   *
   * @param log   the log
   * @param image the image
   * @return the canvas
   */
  public Tensor getCanvas(@Nonnull final NotebookOutput log, final Tensor image) {
    try {
      Tensor canvas = image.copy();
      logger.info(log.image(canvas.toImage(), "Original") + canvas);
      paint(canvas);
      logger.info(log.image(canvas.toImage(), "Corrupted") + canvas);
      return canvas;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Load image tensor.
   *
   * @param img the img
   * @return the tensor
   */
  @Nonnull
  public Tensor loadImage(final String img) {
    try {
      BufferedImage image = ImageIO.read(new File(img));
      image = TestUtil.resize(image, 400, true);
      return Tensor.fromRGB(image);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Paint.
   *
   * @param canvas the canvas
   */
  public void paint(final Tensor canvas) {
    paint_LowRes(canvas, 8);
  }
  
  /**
   * Gets texture network.
   *
   * @param log the log
   * @return the texture network
   * @throws KeyManagementException   the key management exception
   * @throws NoSuchAlgorithmException the no such algorithm exception
   * @throws KeyStoreException        the key store exception
   * @throws IOException              the io exception
   */
  public Layer getTextureNetwork(@Nonnull final NotebookOutput log) throws KeyManagementException, NoSuchAlgorithmException, KeyStoreException, IOException {
    final AtomicReference<Layer> ref = new AtomicReference<>(null);
    new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
      @Override
      protected void phase3() {
        ref.set(pipelineNetwork.copy().freeze());
        super.phase3();
      }
    }.getNetwork();
    return ref.get();
  }
  
  /**
   * The type Layer 2 b.
   */
  public static class Layer2b extends ReproductionDemo {
    /**
     * Instantiates a new Layer 2 b.
     *
     * @param files the files
     */
    public Layer2b(final String... files) {
      super(files);
    }
    
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
      ReproductionDemo demo = new Layer2b(args);
      demo.run(demo::run);
    }
    
    @Override
    public Layer getTextureNetwork(@Nonnull final NotebookOutput log) {
      return log.code(() -> {
        try {
          final AtomicReference<Layer> ref = new AtomicReference<>(null);
          new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
            @Override
            protected void phase2b() {
              ref.set(pipelineNetwork.copy().freeze());
              super.phase2b();
            }
          }.getNetwork();
          return ref.get();
        } catch (@Nonnull final RuntimeException e) {
          throw e;
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
    }
  }
  
  /**
   * The type Layer 2 a.
   */
  public static class Layer2a extends ReproductionDemo {
    
    /**
     * Instantiates a new Layer 2 a.
     *
     * @param files the files
     */
    public Layer2a(final String... files) {
      super(files);
    }
    
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
      ReproductionDemo demo = new Layer2a(args);
      demo.run(demo::run);
    }
    
    @Override
    public Layer getTextureNetwork(@Nonnull final NotebookOutput log) {
      return log.code(() -> {
        try {
          final AtomicReference<Layer> ref = new AtomicReference<>(null);
          new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
            @Override
            protected void phase2a() {
              ref.set(pipelineNetwork.copy().freeze());
              super.phase2a();
            }
          }.getNetwork();
          return ref.get();
        } catch (@Nonnull final RuntimeException e) {
          throw e;
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
    }
  }
}
