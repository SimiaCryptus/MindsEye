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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

public abstract class OptimizerComparison extends ImageTestUtil {
  
  private int timeoutMinutes = 1;
  private final FwdNetworkFactory fwdFactory;
  private final RevNetworkFactory revFactory;
  private final ImageData data;
  
  public OptimizerComparison(FwdNetworkFactory fwdFactory, RevNetworkFactory revFactory, ImageData data) {
    this.fwdFactory = fwdFactory;
    this.revFactory = revFactory;
    this.data = data;
  }
  
  /**
   * Classification comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void classification() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) log.addCopy(originalOut);
      compare(log, opt -> {
        return new ClassifyProblem(fwdFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
      });
    }
  }
  
  /**
   * Encoding comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void encoding() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) log.addCopy(originalOut);
      compare(log, opt -> {
        return new EncodingProblem(revFactory, opt, data)
          .setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
      });
    }
  }
  
  public abstract void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test);
  
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }
  
  public OptimizerComparison setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
}
