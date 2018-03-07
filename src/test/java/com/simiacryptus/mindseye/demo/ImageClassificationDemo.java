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
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
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
public class ImageClassificationDemo extends ArtistryDemo {
  
  /**
   * Instantiates a new Image classification demo.
   *
   * @param args the args
   */
  public ImageClassificationDemo(final String... args) {
  
  }
  
  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    ImageClassificationDemo demo = new ImageClassificationDemo(args);
    demo.run(demo::run);
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    
    
    log.h1("Model");
    ImageClassifier vgg16 = loadModel(log);
  
    log.h1("Data");
    Tensor[] images = loadData(log);
  
    log.h1("Prediction");
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      return vgg16.predict(5, images);
    });
  
    log.h1("Results");
    log.code(() -> {
      @Nonnull TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        @Nonnull HashMap<String, Object> row = new HashMap<>();
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
   * Load data tensor [ ].
   *
   * @param log the log
   * @return the tensor [ ]
   */
  public Tensor[] loadData(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, 224, false);
        return Tensor.fromRGB(img);
      }).limit(10).toArray(i1 -> new Tensor[i1]);
    });
  }
  
  /**
   * Load model image classifier.
   *
   * @param log the log
   * @return the image classifier
   */
  public ImageClassifier loadModel(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      VGG16_HDF5 vgg16_hdf5 = VGG16.fromS3_HDF5();
      vgg16_hdf5.getNetwork();
      return vgg16_hdf5;
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
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return ImageClassifier.class;
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
  
  /**
   * The type Java.
   */
  public static class Java extends ImageClassificationDemo {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
      ImageClassificationDemo demo = new ImageClassificationDemo.Java();
      demo.run(demo::run);
    }
    
    @Override
    public ImageClassifier loadModel(@Nonnull final NotebookOutput log) {
      return log.code(() -> {
        try {
          VGG16_HDF5.JBLAS model = new VGG16_HDF5.JBLAS(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5"))));
          model.getNetwork();
          return model;
        } catch (@Nonnull final RuntimeException e) {
          throw e;
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
    }
    
  }
}
