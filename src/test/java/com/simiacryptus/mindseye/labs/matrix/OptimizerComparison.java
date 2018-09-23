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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;

/**
 * The type Optimizer comparison.
 */
public abstract class OptimizerComparison extends NotebookReportBase {

  /**
   * The Data.
   */
  protected ImageProblemData data;
  /**
   * The Fwd factory.
   */
  protected FwdNetworkFactory fwdFactory;
  /**
   * The Rev factory.
   */
  protected RevNetworkFactory revFactory;
  /**
   * The Timeout minutes.
   */
  protected int timeoutMinutes = 10;

  /**
   * Instantiates a new Optimizer comparison.
   *
   * @param fwdFactory the fwd factory
   * @param revFactory the rev factory
   * @param data       the data
   */
  public OptimizerComparison(final FwdNetworkFactory fwdFactory, final RevNetworkFactory revFactory, final ImageProblemData data) {
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
  public void classification() {
    run(this::classification, getClass().getSimpleName(), "Classification");
  }

  /**
   * Classification.
   *
   * @param log the log
   */
  public void classification(@Nonnull NotebookOutput log) {
    compare(log, opt -> {
      return new ClassifyProblem(fwdFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
    });
  }

  /**
   * Compare.
   *
   * @param log  the log
   * @param test the apply
   */
  public abstract void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test);


  /**
   * Classification comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void encoding() {
    run(this::encoding, getClass().getSimpleName(), "Encoding");
  }

  /**
   * Encoding.
   *
   * @param log the log
   */
  public void encoding(@Nonnull NotebookOutput log) {
    compare(log, opt -> {
      return new EncodingProblem(revFactory, opt, data, 20)
          .setTimeoutMinutes(timeoutMinutes).setTrainingSize(1000).run(log).getHistory();
    });
  }

  /**
   * Gets timeout minutes.
   *
   * @return the timeout minutes
   */
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  /**
   * Sets timeout minutes.
   *
   * @param timeoutMinutes the timeout minutes
   * @return the timeout minutes
   */
  @Nonnull
  public OptimizerComparison setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Experiments;
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return OptimizerComparison.class;
  }
}
