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
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
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
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * The type Deep dream demo.
 */
public class DeepDreamDemo extends ArtistryDemo {
  
  /**
   * Run.
   */
  @Test
  public void run() {
    run(this::run);
  }
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init();
  
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
    
    log.h1("Model");
    DAGNetwork dreamer = log.code(() -> {
      final DAGNetwork[] dreamNet = new DAGNetwork[1];
      try {
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase1d() {
            add(new SquareActivationLayer().setAlpha(-1.0));
            add(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg));
            add(new SumReducerLayer());
            dreamNet[0] = (DAGNetwork) pipelineNetwork.copy();
            throw new RuntimeException("Abort Network Construction");
          }
        }.setLarge(true).setFinalPoolingMode(PoolingLayer.PoolingMode.Avg).getNetwork();
      } catch (@Nonnull final RuntimeException e) {
        // Ignore
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
      return dreamNet[0];
    });

//    Tensor[] images = getImages_Caltech(log);
    Tensor[] images = getImages_Artistry(log);
    
    for (int itemNumber = 0; itemNumber < images.length; itemNumber++) {
      log.h1("Image " + itemNumber);
      Tensor image = images[itemNumber];
      TestUtil.monitorImage(image, false);
      Function<IterativeTrainer, IterativeTrainer> config = train -> train
        .setTimeout(90, TimeUnit.MINUTES)
        .setIterationsPerSample(5);
  
      TestUtil.instrumentPerformance(log, dreamer);
      addLayersHandler(dreamer, server);
      @Nonnull List<Tensor[]> data = Arrays.<Tensor[]>asList(new Tensor[]{image, image.copy()});
      log.code(() -> {
        for (Tensor[] tensors : data) {
          try {
            log.p(log.image(tensors[0].toImage(), ""));
          } catch (IOException e1) {
            throw new RuntimeException(e1);
          }
        }
      });
  
      PipelineNetwork normalized = new PipelineNetwork(2);
      normalized.wrap(dreamer, normalized.getInput(0));
      normalized.wrap(new ProductInputsLayer(), // new BinarySumLayer(0.99,0.1),
        normalized.wrap(dreamer, normalized.getInput(0)),
        normalized.wrap(new NthPowerActivationLayer().setPower(-1),
          normalized.wrap(new LinearActivationLayer().setBias(1000.0).freeze(),
            normalized.wrap(new MeanSqLossLayer(),
              normalized.getInput(1),
              normalized.getInput(0)
            ))));
  
  
      log.code(() -> {
        @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
        @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
        clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
        clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
        @Nonnull PipelineNetwork supervised = new PipelineNetwork(2);
        supervised.add(normalized.freeze(),
          supervised.wrap(clamp, supervised.getInput(0)),
          supervised.getInput(1));
        @Nonnull Trainable trainable = new ArrayTrainable(supervised, 1).setVerbose(true).setMask(true, false).setData(data);
        config.apply(new IterativeTrainer(trainable)
          .setMonitor(ImageClassifier.getTrainingMonitor(history, supervised))
          .setOrientation(new QQN())
          .setLineSearchFactory(name -> new ArmijoWolfeSearch())
          .setTimeout(45, TimeUnit.MINUTES))
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
        return TestUtil.plot(history);
      });
      try {
        log.p(log.image(image.toImage(), "result"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Get images artistry tensor [ ].
   *
   * @param log the log
   * @return the tensor [ ]
   */
  public Tensor[] getImages_Artistry(@Nonnull final NotebookOutput log) {
    return Stream.of(
      "H:\\SimiaCryptus\\Artistry\\Owned\\DSC_0005.JPG",
      "H:\\SimiaCryptus\\Artistry\\Owned\\DSC_0127.JPG",
      "H:\\SimiaCryptus\\Artistry\\Owned\\DSC00097.JPG",
      "H:\\SimiaCryptus\\Artistry\\Owned\\DSC00152.JPG",
      "H:\\SimiaCryptus\\Artistry\\Owned\\DSC00200.JPG",
      "H:\\SimiaCryptus\\Artistry\\monkeydog.jpg",
      "H:\\SimiaCryptus\\Artistry\\landscape.jpg",
      "H:\\SimiaCryptus\\Artistry\\chimps\\winner.jpg",
      "H:\\SimiaCryptus\\Artistry\\chimps\\working.jpg",
      "H:\\SimiaCryptus\\Artistry\\autumn.jpg"
    ).map(file -> {
      try {
        BufferedImage image = ImageIO.read(new File(file));
        image = TestUtil.resize(image, 600, true);
        return Tensor.fromRGB(image);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Get images caltech tensor [ ].
   *
   * @param log the log
   * @return the tensor [ ]
   */
  public Tensor[] getImages_Caltech(@Nonnull final NotebookOutput log) {
    log.h1("Data");
    return log.code(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        //img = TestUtil.resize(img, 224);
        return Tensor.fromRGB(img);
      }).limit(50).toArray(i1 -> new Tensor[i1]);
    });
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
