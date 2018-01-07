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

import com.simiacryptus.mindseye.labs.encoding.EncodingUtil;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.SerializationTest;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import java.util.*;

/**
 * The type Image classifier run base.
 */
public abstract class ImageClassifierTestBase extends NotebookReportBase {
  
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
  public void run(NotebookOutput log) {
    ImageClassifier vgg16 = getImageClassifier(log);
    NNLayer network = ((DemoableNetworkFactory) vgg16).build(log);

    log.h1("Network Diagram");
    log.p("This is a diagram of the imported network:");
    log.code(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) network))
                     .height(4000).width(800).render(Format.PNG).toImage();
    });
  
    SerializationTest serializationTest = new SerializationTest();
    serializationTest.test(log, network, (Tensor[]) null);

    log.h1("Predictions");
    Tensor[][] images = EncodingUtil.getImages(log, 224, 10);
    Map<String, List<LinkedHashMap<String, Double>>> modelPredictions = new HashMap<>();
    modelPredictions.put("Source", predict(log, vgg16, network, images));
    serializationTest.getModels().forEach((precision, model) -> {
      log.h2(precision.name());
      modelPredictions.put(precision.name(), predict(log, vgg16, model, images));
    });
  
    log.h1("Result");

    log.code(() -> {
      TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        int index = i;
        HashMap<String, Object> row = new HashMap<>();
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
  public List<LinkedHashMap<String, Double>> predict(NotebookOutput log, ImageClassifier vgg16, NNLayer network, Tensor[][] images) {
    TestUtil.instrumentPerformance(log, (DAGNetwork) network);
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      Tensor[] data = Arrays.stream(images).map(x -> x[1]).toArray(i -> new Tensor[i]);
      return ImageClassifier.predict(
        vgg16::prefilter, network, 5, vgg16.getCategories(), 1, data);
    });
    TestUtil.extractPerformance(log, (DAGNetwork) network);
    return predictions;
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected abstract Class<?> getTargetClass();
  
  @Override
  public ReportType getReportType() {
    return ReportType.Models;
  }
}