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
import com.simiacryptus.mindseye.test.NotebookOutputTestBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.JsonTest;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.CodeUtil;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import java.util.*;

/**
 * The type Image classifier test base.
 */
public abstract class ImageClassifierTestBase extends NotebookOutputTestBase {
  
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void test() throws Throwable {
    try (NotebookOutput log = getLog()) {
      printHeader(log);
      ImageClassifier imageClassifier = getImageClassifier(log);
      test(log, imageClassifier);
    }
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
   * @param log   the log
   * @param vgg16 the vgg 16
   */
  public void test(NotebookOutput log, ImageClassifier vgg16) {
    //log.p("__Test Description:__ " + testJavadoc);
    
    PipelineNetwork network = vgg16.build(log);
    log.h1("Network Diagram");
    log.p("This is a diagram of the imported network:");
    log.code(() -> {
      return Graphviz.fromGraph(TestUtil.toGraph(network))
                     .height(4000).width(800).render(Format.PNG).toImage();
    });
    log.h1("Predictions");
    Tensor[][] images = EncodingUtil.getImages(log, 224, 10);
    TestUtil.instrumentPerformance(log, network);
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      return vgg16.predict(5, Arrays.stream(images).map(x -> x[1]).toArray(i -> new Tensor[i]));
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
    log.h1("JSON");
    new JsonTest().test(log, network, null);
  }
  
  /**
   * Print header.
   *
   * @param log the log
   */
  public void printHeader(NotebookOutput log) {
    Class<?> networkClass = getTargetClass();
    Class<? extends NotebookOutputTestBase> selfClass = getClass();
    String appJavadoc = CodeUtil.getJavadoc(networkClass);
    String testJavadoc = CodeUtil.getJavadoc(selfClass);
    log.setFrontMatterProperty("created_on", new Date().toString());
    log.setFrontMatterProperty("application_class_short", networkClass.getSimpleName());
    log.setFrontMatterProperty("application_class_full", networkClass.getCanonicalName());
    log.setFrontMatterProperty("application_class_doc", appJavadoc.replaceAll("\n", ""));
    log.setFrontMatterProperty("test_class_short", selfClass.getSimpleName());
    log.setFrontMatterProperty("test_class_full", selfClass.getCanonicalName());
    log.setFrontMatterProperty("test_class_doc", testJavadoc.replaceAll("\n", ""));
    log.p("__Application Description:__ " + appJavadoc);
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected abstract Class<?> getTargetClass();
}
