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
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * We load a pretrained convolutional neural network (VGG16) along with the CalTech101 image dataset to perform a
 * demonstration of Image Recognition.
 */
public class ImageClassificationDemo extends NotebookReportBase {
  
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
  
  
    log.h1("Model");
    
    ImageClassifier vgg16 = log.code(() -> {
      return VGG16.fromS3_HDF5();
    });
  
    log.h1("Data");
    Tensor[] images = log.code(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, 224);
        return Tensor.fromRGB(img);
      }).limit(10).toArray(i1 -> new Tensor[i1]);
    });
  
    log.h1("Prediction");
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      return vgg16.predict(5, images);
    });
  
    log.h1("Results");
    log.code(() -> {
      @javax.annotation.Nonnull TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        @javax.annotation.Nonnull HashMap<String, Object> row = new HashMap<>();
        row.put("Image", log.image(images[i].toImage(), ""));
        row.put("Prediction", predictions.get(i).entrySet().stream()
          .map(e -> String.format("%s -> %.2f", e.getKey(), 100 * e.getValue()))
          .reduce((a, b) -> a + "<br/>" + b).get());
        tableOutput.putRow(row);
      }
      return tableOutput;
    }, 256 * 1024);
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
    return ImageClassifier.class;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
}
