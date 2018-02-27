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
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The type Image classifier run base.
 */
public class TextureDemo extends ArtistryDemo {
  
  
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
  
    Layer textureNetork = loadTextureNetwork(log);
    Tensor textureVector = getFeatureVector(log);
    Tensor rendering = new Tensor(300, 300, 3);
    paint_Lines(rendering);
    TestUtil.monitorImage(rendering, false, 5);
  
    @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
    clamp.add(new ActivationLayer(ActivationLayer.Mode.RELU));
    clamp.add(new LinearActivationLayer().setBias(255).setScale(-1).freeze());
  
    log.code(() -> {
      @Nonnull PipelineNetwork painterNetwork = new PipelineNetwork(2);
      painterNetwork.wrap(new MeanSqLossLayer(),
        painterNetwork.add(new BandReducerLayer(),
          painterNetwork.add(textureNetork,
            painterNetwork.add(clamp,
              painterNetwork.getInput(0)))),
        painterNetwork.getInput(1));
    
      BasicTrainable trainable = new BasicTrainable(painterNetwork);
      trainable.setMask(true, false);
      trainable.setData(Arrays.<Tensor[]>asList(new Tensor[]{rendering, textureVector}));
    
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setOrientation(new QQN())
        .setLineSearchFactory(name -> new QuadraticSearch().setCurrentRate(20).setRelativeTolerance(0.05))
        .setTimeout(4, TimeUnit.HOURS)
        .runAndFree();
      return TestUtil.plot(history);
    });
  
    try {
      logger.info(log.image(rendering.toImage(), "Rendering"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  
    log.setFrontMatterProperty("status", "OK");
  }
  
  protected Tensor getFeatureVector(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      return new Tensor(1, 1, 4096).setAll(0.0).set(5, 1.0);
    });
  }
  
  protected Layer loadTextureNetwork(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      try {
        final AtomicReference<Layer> ref = new AtomicReference<>(null);
        new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3() {
            ref.set(pipelineNetwork.copy().freeze());
            super.phase3();
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
  
  public static class Layer2b extends TextureDemo {
    @Override
    protected Tensor getFeatureVector(@Nonnull final NotebookOutput log) {
      return log.code(() -> {
        return new Tensor(1, 1, 512).setAll(0.0).set(5, 1.0);
      });
    }
    
    @Override
    protected Layer loadTextureNetwork(@Nonnull final NotebookOutput log) {
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
}
