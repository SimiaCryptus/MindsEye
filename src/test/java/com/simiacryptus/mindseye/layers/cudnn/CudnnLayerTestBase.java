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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;

/**
 * The type Cudnn layer run base.
 */
public abstract class CudnnLayerTestBase extends LayerTestBase {
  
  /**
   * Instantiates a new Cudnn layer run base.
   */
  public CudnnLayerTestBase() {
  }
  
  @Override
  public void run(NotebookOutput log) {
    super.run(log);
    log.p("CuDNN Statistics:");
    log.code(() -> {
      return TestUtil.toFormattedJson(CuDNN.getExecutionStatistics());
    });
  }
  
  @Override
  protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
    final ComponentTest<ToleranceStatistics> inner = super.getReferenceIOTester();
    return (log, component, inputPrototype) -> {
      PrintStream apiLog = null;
      try {
        String logName = "cuda_" + log.getName() + "_io.log";
        log.p(log.file((String) null, logName, "GPU Log"));
        apiLog = new PrintStream(log.file(logName));
        CuDNN.addLog(apiLog);
        return inner.test(log, component, inputPrototype);
      } finally {
        if (null != apiLog) {
          apiLog.close();
          CuDNN.apiLog.remove(apiLog);
        }
      }
    };
  }
  
  @Override
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    ComponentTest<ToleranceStatistics> inner = super.getPerformanceTester();
    return (log, component, inputPrototype) -> {
      PrintStream apiLog = null;
      try {
        String logName = "cuda_" + log.getName() + "_perf.log";
        log.p(log.file((String) null, logName, "GPU Log"));
        apiLog = new PrintStream(log.file(logName));
        CuDNN.addLog(apiLog);
        return inner.test(log, component, inputPrototype);
      } finally {
        if (null != apiLog) {
          apiLog.close();
          CuDNN.apiLog.remove(apiLog);
        }
      }
    };
  }
}
