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

import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * The type ArtistryAppBase demo.
 */
public abstract class ArtistryAppBase extends NotebookReportBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ArtistryAppBase.class);
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public final void run() {
    run(notebookOutput -> run(notebookOutput), getClass().getSimpleName() + "_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
  /**
   * Run.
   *
   * @param notebookOutput the notebook output
   */
  protected abstract void run(final NotebookOutput notebookOutput);
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }
  
  /**
   * Init.
   *
   * @param log the log
   */
  public void init(final NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    //server.dataReciever
    //server.init();
    //server.start();
//    @Nonnull String logName = "cuda_" + log.getName() + ".log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    CudaSystem.addLog(new PrintStream(log.file(logName)));
  }
  
}
