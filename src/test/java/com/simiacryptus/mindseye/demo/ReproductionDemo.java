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
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
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
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image classifier run base.
 */
public class ReproductionDemo extends ArtistryDemo {
  
  
  /**
   * The Texture netork.
   */
  
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
   * @param log the _log
   */
  public void run(@Nonnull NotebookOutput log) {
  
    @Nonnull String logName = "cuda_" + log.getName() + "._log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
    
    log.h1("Model");
    Layer textureNetork = log.code(() -> {
      try {
        final AtomicReference<Layer> ref = new AtomicReference<>(null);
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3() {
            ref.set(model.copy().freeze());
            super.phase3();
          }
        }.build();
        return ref.get();
      } catch (@Nonnull final RuntimeException e) {
        throw e;
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
    });
  
    Tensor[][] inputData = Stream.of("H:\\SimiaCryptus\\Artistry\\chimps\\chip.jpg").map(img -> {
      try {
        BufferedImage image = ImageIO.read(new File(img));
        image = TestUtil.resize(image, 400, true);
        return new Tensor[]{Tensor.fromRGB(image)};
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).toArray(i -> new Tensor[i][]);
    
    Tensor[][] preprocessedTrainingData = IntStream.range(0, inputData.length).mapToObj(i -> {
      TimedResult<Tensor[]> timedResult = TimedResult.time(() -> {
        Tensor image = inputData[i][0];
        return new Tensor[]{textureNetork.eval(image).getDataAndFree().getAndFree(0), image};
      });
      logger.info(String.format("Preprocessed record %d/%d in %.3f", i, inputData.length, timedResult.seconds()));
      return timedResult.result;
    }).toArray(i -> new Tensor[i][]);
    
    log.h1("Model Training");
    
    log.h1("Output Processing");
    for (int itemNumber = 0; itemNumber < preprocessedTrainingData.length; itemNumber++) {
      log.h1("Image " + itemNumber);
      @Nonnull List<Tensor[]> row = Arrays.<Tensor[]>asList(preprocessedTrainingData[itemNumber]);
  
      //TestUtil.monitorImage(preprocessedTrainingData[itemNumber][1].copy(), false, Integer.MAX_VALUE);
      TestUtil.monitorImage(preprocessedTrainingData[itemNumber][1], false, 5);
      log.code(() -> {
        for (Tensor[] tensors : row) {
          try {
            Tensor canvas = tensors[1];
            logger.info(log.image(canvas.toImage(), "Original") + canvas);
            paint_LowRes(canvas);
            logger.info(log.image(canvas.toImage(), "Corrupted") + canvas);
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      });
      
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      
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
      BasicTrainable basicTrainable = new BasicTrainable(painterNetwork);
      basicTrainable.setMask(false, true);
      basicTrainable.setData(row);
      new IterativeTrainer(basicTrainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new QuadraticSearch().setCurrentRate(20).setRelativeTolerance(0.05))
        .setTimeout(8, TimeUnit.HOURS)
        .runAndFree();
      log.code(() -> {
        return TestUtil.plot(history);
      });
      
      log.code(() -> {
        for (Tensor[] tensors : row) {
          try {
            logger.info(log.image(tensors[1].toImage(), "Reproduction"));
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      });
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
}
