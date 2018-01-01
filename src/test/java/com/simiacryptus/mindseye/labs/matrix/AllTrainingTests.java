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
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;

/**
 * The type All training tests.
 */
public abstract class AllTrainingTests extends NotebookReportBase {
  /**
   * The Fwd factory.
   */
  protected final FwdNetworkFactory fwdFactory;
  /**
   * The Optimization strategy.
   */
  protected final OptimizationStrategy optimizationStrategy;
  /**
   * The Rev factory.
   */
  protected final RevNetworkFactory revFactory;
  /**
   * The Timeout minutes.
   */
  protected int timeoutMinutes = 10;
  /**
   * The Batch size.
   */
  protected int batchSize = 1000;
  
  /**
   * Instantiates a new All training tests.
   *
   * @param fwdFactory           the fwd factory
   * @param revFactory           the rev factory
   * @param optimizationStrategy the optimization strategy
   */
  public AllTrainingTests(final FwdNetworkFactory fwdFactory, final RevNetworkFactory revFactory, final OptimizationStrategy optimizationStrategy) {
    this.fwdFactory = fwdFactory;
    this.revFactory = revFactory;
    this.optimizationStrategy = optimizationStrategy;
  }
  
  /**
   * Autoencoder test.
   *
   * @param log the log
   */
  public void autoencoder_test(NotebookOutput log) {
    log.h1(getDatasetName() + " Denoising Autoencoder");
    intro(log);
    new AutoencodingProblem(fwdFactory, optimizationStrategy, revFactory, getData(), 100, 0.8).setTimeoutMinutes(timeoutMinutes).run(log);
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Training;
  }
  
  /**
   * Autoencoder run.
   *
   * @throws IOException the io exception
   */
  @Test
  @Ignore
  @Category(TestCategories.Report.class)
  public void autoencoder_test() throws IOException {
    run(this::autoencoder_test, getClass().getSimpleName(), "Autoencoder");
  }
  
  /**
   * Classification test.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void classification_test() throws IOException {
    run(this::classification_test, getClass().getSimpleName(), "Classification");
  }
  
  /**
   * Classification test.
   *
   * @param log the log
   */
  public void classification_test(NotebookOutput log) {
    log.h1(getDatasetName() + " Denoising Autoencoder");
    intro(log);
    new ClassifyProblem(fwdFactory, optimizationStrategy, getData(), 100).setBatchSize(batchSize).setTimeoutMinutes(timeoutMinutes).run(log);
  }
  
  /**
   * Encoding test.
   *
   * @throws IOException the io exception
   */
  @Test
  @Ignore
  @Category(TestCategories.Report.class)
  public void encoding_test() throws IOException {
    run(this::encoding_test, getClass().getSimpleName(), "Encoding");
  }
  
  /**
   * Encoding test.
   *
   * @param log the log
   */
  public void encoding_test(NotebookOutput log) {
    log.h1(getDatasetName() + " Image-to-Vector Encoding");
    intro(log);
    new EncodingProblem(revFactory, optimizationStrategy, getData(), 10).setTimeoutMinutes(timeoutMinutes).run(log);
  }
  
  @Override
  public void printHeader(NotebookOutput log) {
    String fwdFactory_javadoc = printHeader(log, fwdFactory.getClass(), "fwd");
    String optimizationStrategy_javadoc = printHeader(log, optimizationStrategy.getClass(), "opt");
    String revFactory_javadoc = printHeader(log, revFactory.getClass(), "rev");
    super.printHeader(log);
    log.p("_Forward Strategy Javadoc_: " + fwdFactory_javadoc);
    log.p("_Reverse Strategy Javadoc_: " + revFactory_javadoc);
    log.p("_Optimization Strategy Javadoc_: " + optimizationStrategy_javadoc);
  }
  
  /**
   * Intro.
   *
   * @param log the log
   */
  protected abstract void intro(NotebookOutput log);
  
  /**
   * Gets data.
   *
   * @return the data
   */
  public abstract ImageProblemData getData();
  
  /**
   * Gets dataset name.
   *
   * @return the dataset name
   */
  public abstract String getDatasetName();
}
