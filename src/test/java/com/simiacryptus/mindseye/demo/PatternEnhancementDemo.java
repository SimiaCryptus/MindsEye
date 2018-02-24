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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.ImgPixelSoftmaxLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Comparator;

/**
 * The type Image classifier run base.
 */
public class PatternEnhancementDemo extends NotebookReportBase {
  
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
    
    
    @javax.annotation.Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
    
    log.h1("Model");
    VGG16_HDF5 vgg16 = log.code(() -> {
      final VGG16_HDF5 result;
      try {
        result = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3b(@Nonnull final NotebookOutput output) {
            output.code(() -> {
              PipelineNetwork imageFitness = new PipelineNetwork(1);
              InnerNode softmax = imageFitness.wrap(new ImgPixelSoftmaxLayer(), imageFitness.getInput(0));
              imageFitness.wrap(new EntropyLossLayer(), softmax, softmax); // Minimize self-entropy to enhance classifyable patterns
              imageFitness.wrap(new SumReducerLayer());
              add(imageFitness);
            });
          }
        };
      } catch (@javax.annotation.Nonnull final RuntimeException e) {
        throw e;
      } catch (Throwable e) {
        throw new RuntimeException(e);
      }
      return result.setLarge(false);
    });
    
    log.h1("Data");
    Tensor imageData;
    try {
      BufferedImage image = ImageIO.read(new File("H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\800px-Vincent_van_Gogh_-_Portrait_of_Doctor_Félix_Rey_(F500).jpg"));
      image = TestUtil.resize(image, 400, true);
      imageData = Tensor.fromRGB(image);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    TestUtil.monitorImage(imageData, false);
    vgg16.deepDream(log, imageData);
    try {
      log.p(log.image(imageData.toImage(), "result"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    
    log.setFrontMatterProperty("status", "OK");
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
