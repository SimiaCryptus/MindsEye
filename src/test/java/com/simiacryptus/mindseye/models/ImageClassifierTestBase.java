/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.JsonTest;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

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
    super.run();
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
   *  @param log   the log
   *
   */
  public void run(NotebookOutput log) {
    ImageClassifier vgg16 = getImageClassifier(log);
    PipelineNetwork network = vgg16.build(log);

    log.h1("Network Diagram");
    log.p("This is a diagram of the imported network:");
    log.code(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(network))
                     .height(4000).width(800).render(Format.PNG).toImage();
    });
  
    new JsonTest().test(log, network, (Tensor[]) null);
  
    log.h1("Predictions");
    Tensor[][] images = EncodingUtil.getImages(log, 224, 10);
    TestUtil.instrumentPerformance(log, network);
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      Tensor[] data = Arrays.stream(images).map(x -> x[1]).toArray(i -> new Tensor[i]);
      return ImageClassifier.predict(
        vgg16::prefilter, network, 5, vgg16.getCategories(), data);
    });
    TestUtil.extractPerformance(log, network);
    log.code(() -> {
      TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        HashMap<String, Object> row = new HashMap<>();
        row.put("Image", log.image(images[i][1].toImage(), ""));
        row.put("Prediction", predictions.get(i).entrySet().stream()
                                         .map(e -> String.format("%s -> %.2f", e.getKey(), 100 * e.getValue()))
                                         .reduce((a, b) -> a + ";" + b).get());
        tableOutput.putRow(row);
      }
      return tableOutput;
    }, 256 * 1024);
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected abstract Class<?> getTargetClass();
}
