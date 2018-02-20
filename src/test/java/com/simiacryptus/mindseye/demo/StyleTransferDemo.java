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
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.orient.RecursiveSubspace;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.ImageTiles;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image classifier run base.
 */
public class StyleTransferDemo extends NotebookReportBase {
  
  
  Layer textureNetork;
  
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
  public void run(@javax.annotation.Nonnull NotebookOutput log) {
  
    List<String> control = Arrays.asList("H:\\SimiaCryptus\\Artistry\\portraits\\photos");
    List<String> target = Arrays.asList("H:\\SimiaCryptus\\Artistry\\portraits\\picasso");
    String input = "H:\\SimiaCryptus\\Artistry\\portraits\\photos\\1280px-Winter_baby_10-months-old.jpg";
  
    @javax.annotation.Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
  
    log.h1("Model");
    PipelineNetwork trainedCategorizer = buildCategorizer();
    Layer fullNetwork = log.code(() -> {
      try {
        return new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3(@Nonnull final NotebookOutput output) {
            output.code(() -> {
              textureNetork = model.copy().freeze();
              add(trainedCategorizer);
            });
          }
        };
      } catch (@Nonnull final RuntimeException e) {
        throw e;
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
    }).getNetwork().freeze();
    //textureNetork = new RescaledSubnetLayer(2,textureNetork);
  
    Tensor[][] rawTrainingData = Stream.concat(
      control.stream().flatMap(f -> loadTiles(f, 1.0, 0.0)),
      target.stream().flatMap(f -> loadTiles(f, 0.0, 1.0))
    ) //
      .limit(10)
      .toArray(i -> new Tensor[i][]);
  
    Tensor[][] inputData = Stream.concat(
      Stream.of(input).map(img -> {
        try {
          return new Tensor[]{Tensor.fromRGB(ImageIO.read(new File(img))), new Tensor(1.0, 0.0)};
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }),
      Stream.empty()
    ) //
      .flatMap(ts -> ImageTiles.toTiles(ts[0].toImage(), 250, 250, 250, 250, Integer.MAX_VALUE, Integer.MAX_VALUE).stream().map(t -> new Tensor[]{t, ts[1]}))
      .toArray(i -> new Tensor[i][]);
  
    Tensor[][] preprocessedTrainingData = IntStream.range(0, rawTrainingData.length).mapToObj(i -> {
      Tensor[] x = rawTrainingData[i];
      TimedResult<Tensor[]> timedResult = TimedResult.time(() -> new Tensor[]{textureNetork.eval(x[0]).getDataAndFree().getAndFree(0), x[1]});
      logger.info(String.format("Preprocessed record %d/%d in %.3f", i, rawTrainingData.length, timedResult.seconds()));
      return timedResult.result;
    }).toArray(i -> new Tensor[i][]);
  
    log.h1("Model Training");
  
    @javax.annotation.Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    @javax.annotation.Nonnull PipelineNetwork supervised1 = new PipelineNetwork(2);
    supervised1.wrap(new EntropyLossLayer(),
      supervised1.add(trainedCategorizer, supervised1.getInput(0)),
      supervised1.getInput(1));
    new IterativeTrainer(new SampledArrayTrainable(preprocessedTrainingData, supervised1, 10, 1))
      .setMonitor(getTrainingMonitor(history))
      .setOrientation(new RecursiveSubspace())
      .setLineSearchFactory(name -> new ArmijoWolfeSearch())
      .setTimeout(60, TimeUnit.MINUTES)
      .runAndFree();
    log.code(() -> {
      return TestUtil.plot(history);
    });
    fullNetwork.freeze();
  
    log.h1("Output Processing");
    for (int itemNumber = 0; itemNumber < inputData.length; itemNumber++) {
      log.h1("Image " + itemNumber);
      @javax.annotation.Nonnull List<Tensor[]> data = Arrays.<Tensor[]>asList(inputData[itemNumber]);
      log.code(() -> {
        for (Tensor[] tensors : data) {
          try {
            logger.info(log.image(tensors[0].toImage(), "") + tensors[1]);
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      });
      history.clear();
      @javax.annotation.Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
      @javax.annotation.Nonnull PipelineNetwork supervised2 = new PipelineNetwork(2);
      supervised2.wrap(new EntropyLossLayer(),
        supervised2.add(fullNetwork, supervised2.wrap(clamp, supervised2.getInput(0))),
        supervised2.getInput(1));
      @javax.annotation.Nonnull Trainable trainable = new ArrayTrainable(supervised2, 1).setMask(true, false).setData(data);
      new IterativeTrainer(trainable)
        .setMonitor(getTrainingMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(15, TimeUnit.MINUTES)
        .runAndFree();
      log.code(() -> {
        return TestUtil.plot(history);
      });
    
      try {
        log.p(log.image(inputData[itemNumber][0].toImage(), "result"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  
    log.setFrontMatterProperty("status", "OK");
  }
  
  private Stream<Tensor[]> loadTiles(final String f, final double... category) {
    return Arrays.stream(new File(f).listFiles()).flatMap(img -> {
      BufferedImage image;
      try {
        image = ImageIO.read(img);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return ImageTiles.toTiles(image, 250, 250, 250, 250, Integer.MAX_VALUE, Integer.MAX_VALUE).stream().map(t -> {
        return new Tensor[]{t, new Tensor(category)};
      });
    });
  }
  
  @Nonnull
  private PipelineNetwork buildCategorizer() {
    PipelineNetwork trainedCategorizer = new PipelineNetwork();
    trainedCategorizer.add(new ConvolutionLayer(1, 1, 4096, 4096)
      .setPaddingXY(0, 0).setWeightsLog(-4));
    trainedCategorizer.add(new ImgBandBiasLayer(4096).setWeightsLog(-4));
    trainedCategorizer.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    trainedCategorizer.add(new ConvolutionLayer(1, 1, 4096, 2)
      .setPaddingXY(0, 0).setWeightsLog(-4));
    trainedCategorizer.add(new ImgBandBiasLayer(2).setWeightsLog(-4));
    trainedCategorizer.add(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg));
    trainedCategorizer.add(new SoftmaxActivationLayer());
    return trainedCategorizer;
  }
  
  /**
   * Gets training monitor.
   *
   * @param history the history
   * @return the training monitor
   */
  @javax.annotation.Nonnull
  public TrainingMonitor getTrainingMonitor(@javax.annotation.Nonnull ArrayList<StepRecord> history) {
    @javax.annotation.Nonnull TrainingMonitor monitor = TestUtil.getMonitor(history);
    return new TrainingMonitor() {
      @Override
      public void clear() {
        monitor.clear();
      }
      
      @Override
      public void log(String msg) {
        SysOutInterceptor.ORIGINAL_OUT.println(msg);
        monitor.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        monitor.onStepComplete(currentPoint);
      }
    };
  }
  
  /**
   * Gets shuffle comparator.
   *
   * @param <T> the type parameter
   * @return the shuffle comparator
   */
  public <T> Comparator<T> getShuffleComparator() {
    final int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return Comparator.comparingInt(a1 -> System.identityHashCode(a1) ^ seed);
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @javax.annotation.Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
}
