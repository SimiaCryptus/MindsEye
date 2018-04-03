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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.mindseye.applications.ImageClassifier;
import com.simiacryptus.mindseye.labs.encoding.EncodingUtil;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The type Image classifier apply base.
 */
public abstract class ImageClassifierTestBase extends NotebookReportBase {
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test(timeout = 30 * 60 * 60 * 1000)
  public void run() {
    run(this::run);
  }
  
  /**
   * Gets image classifier.
   *
   * @param log the log
   * @return the image classifier
   */
  public abstract ImageClassifier getImageClassifier(NotebookOutput log);
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    Future<Tensor[][]> submit = Executors.newSingleThreadExecutor()
      .submit(() -> Arrays.stream(EncodingUtil.getImages(log, img -> {
        return img;
//        return TestUtil.resize(img, 224, 224);
//        if(img.getWidth()>img.getHeight()) {
//          return TestUtil.resize(img, 224, img.getHeight() * 224 / img.getWidth());
//        } else {
//          return TestUtil.resize(img, img.getWidth() * 224 / img.getHeight(), 224);
//        }
      }, 10, new CharSequence[]{}))
        .toArray(i -> new Tensor[i][]));
    ImageClassifier vgg16 = getImageClassifier(log);
    @Nonnull Layer network = vgg16.getNetwork();
    
    log.h1("Network Diagram");
    log.p("This is a diagram of the imported network:");
    log.code(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) network))
        .height(4000).width(800).render(Format.PNG).toImage();
    });

//    @javax.annotation.Nonnull SerializationTest serializationTest = new SerializationTest();
//    serializationTest.setPersist(true);
//    serializationTest.test(log, network, (Tensor[]) null);
  
    log.h1("Predictions");
    Tensor[][] images;
    try {
      images = submit.get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    @Nonnull Map<CharSequence, List<LinkedHashMap<CharSequence, Double>>> modelPredictions = new HashMap<>();
    modelPredictions.put("Source", predict(log, vgg16, network, images));
    network.freeRef();
//    serializationTest.getModels().forEach((precision, model) -> {
//      log.h2(precision.name());
//      modelPredictions.put(precision.name(), predict(log, vgg16, model, images));
//    });
  
    log.h1("Result");
  
    log.code(() -> {
      @Nonnull TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        int index = i;
        @Nonnull HashMap<CharSequence, Object> row = new HashMap<>();
        row.put("Image", log.image(images[i][1].toImage(), ""));
        modelPredictions.forEach((model, predictions) -> {
          row.put(model, predictions.get(index).entrySet().stream()
            .map(e -> String.format("%s -> %.2f", e.getKey(), 100 * e.getValue()))
            .reduce((a, b) -> a + "<br/>" + b).get());
          
        });
        tableOutput.putRow(row);
      }
      return tableOutput;
    }, 256 * 1024);

//    log.p("CudaSystem Statistics:");
//    log.code(() -> {
//      return TestUtil.toFormattedJson(CudaSystem.getExecutionStatistics());
//    });
  
  }
  
  /**
   * Predict list.
   *
   * @param log     the log
   * @param vgg16   the vgg 16
   * @param network the network
   * @param images  the images
   * @return the list
   */
  public List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull NotebookOutput log, @Nonnull ImageClassifier vgg16, @Nonnull Layer network, @Nonnull Tensor[][] images) {
    TestUtil.instrumentPerformance((DAGNetwork) network);
    List<LinkedHashMap<CharSequence, Double>> predictions = log.code(() -> {
      Tensor[] data = Arrays.stream(images).map(x -> x[1]).toArray(i -> new Tensor[i]);
      return ImageClassifier.predict(network, 5, vgg16.getCategories(), 1, data);
    });
    TestUtil.extractPerformance(log, (DAGNetwork) network);
    return predictions;
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected abstract Class<?> getTargetClass();
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Models;
  }
}
