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

package com.simiacryptus.mindseye.applications;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * The type Image classification base.
 */
public abstract class ImageClassificationBase extends ArtistryAppBase {
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    log.h1("Model");
    log.p("In this demonstration, we will show how to load an png recognition network and use it to identify object in images.");
    log.p(
      "We start by loading the VGG16 pretrained model using the HD5 importer. This downloads, if needed, the weights from a file in S3 and re-constructs the network architecture by custom run.");
    log.p("Next, we need an example png to analyze:");
    log.p(
      "We pass this png to the categorization network, and get the following top-10 results. Note that multiple objects may be detected, and the total percentage may be greater than 100%.");
    log.p(
      "Once we have categories identified, we can attempt to localize each object category within the png. We do this via a pipeline starting with the backpropagated input signal evalInputDelta and applying several filters e.g. blurring and normalization to produce an alphaList channel. When applied to the input png, we highlight the png areas related to the object type in question. Note that this produces a fuzzy blob, which does indicate object location but is a poor indicator of object boundaries. Below we perform this task for the top 5 object categories:");
    ImageClassifier vgg16 = loadModel(log);
    
    log.h1("Data");
    Tensor[] images = loadData(log);
    
    log.h1("Prediction");
    List<LinkedHashMap<CharSequence, Double>> predictions = log.eval(() -> {
      return vgg16.predict(5, images);
    });
    
    log.h1("Results");
    log.out(() -> {
      @Nonnull TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        @Nonnull HashMap<CharSequence, Object> row = new HashMap<>();
        row.put("Image", log.png(images[i].toImage(), ""));
        row.put("Prediction", predictions.get(i).entrySet().stream()
          .map(e -> String.format("%s -> %.2f", e.getKey(), 100 * e.getValue()))
          .reduce((a, b) -> a + "<br/>" + b).get());
        tableOutput.putRow(row);
      }
      return tableOutput;
    });
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Load data tensor [ ].
   *
   * @param log the log
   * @return the tensor [ ]
   */
  public Tensor[] loadData(@Nonnull final NotebookOutput log) {
    return log.eval(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, 224, false);
        return Tensor.fromRGB(img);
      }).limit(10).toArray(i1 -> new Tensor[i1]);
    });
  }
  
  /**
   * Load model png classifier.
   *
   * @param log the log
   * @return the png classifier
   */
  public abstract ImageClassifier loadModel(@Nonnull NotebookOutput log);
  
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
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }
}
