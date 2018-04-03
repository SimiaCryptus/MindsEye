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

package com.simiacryptus.mindseye.app;

import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.hadoop.yarn.webapp.MimeType;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.PrintStream;
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
    run(this::run, getClass().getSimpleName() + "_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
  protected abstract void run(final NotebookOutput notebookOutput);
  
  /**
   * The Server.
   */
  protected StreamNanoHTTPD server;
  
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
    try {
      server = new StreamNanoHTTPD(9090).init();
      server.addSyncHandler("gpu.json", MimeType.JSON, out -> {
        try {
          JsonUtil.MAPPER.writer().writeValue(out, CudaSystem.getExecutionStatistics());
          //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
          out.close();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }, false);
      //server.dataReciever
      //server.init();
      //server.start();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
  }
}
