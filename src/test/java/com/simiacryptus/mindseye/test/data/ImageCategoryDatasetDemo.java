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

package com.simiacryptus.mindseye.test.data;

import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.SupplierWeakCache;
import com.simiacryptus.util.test.LabeledObject;
import org.junit.Test;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Image category dataset demo.
 */
public abstract class ImageCategoryDatasetDemo extends NotebookReportBase {
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
  public void run(NotebookOutput log) {
    log.h3("Loading Data");
    List<LabeledObject<SupplierWeakCache<BufferedImage>>> testData =
      getTrainingStream(log).sorted(getShuffleComparator()).collect(Collectors.toList());
    
    log.h3("Categories");
    log.code(() -> {
      testData.stream().collect(Collectors.groupingBy(x -> x.label, Collectors.counting()))
              .forEach((k, v) -> logger.info(String.format("%s -> %d", k, v)));
    });
    
    log.h3("Sample Data");
    log.code(() -> {
      return testData.stream().map(labeledObj -> {
        try {
          BufferedImage img = labeledObj.data.get();
          img = TestUtil.resize(img, 224, true);
          return log.image(img, labeledObj.label);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).limit(20).reduce((a, b) -> a + b).get();
    }, 256 * 1024);
  }
  
  /**
   * Gets training stream.
   *
   * @param log the log
   * @return the training stream
   */
  public abstract Stream<LabeledObject<SupplierWeakCache<BufferedImage>>> getTrainingStream(NotebookOutput log);
  
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
  protected Class<?> getTargetClass() {
    return VGG16_HDF5.class;
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Data;
  }
}
