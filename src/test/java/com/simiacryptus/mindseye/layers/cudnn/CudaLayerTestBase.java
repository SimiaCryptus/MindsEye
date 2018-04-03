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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.ComponentTestBase;
import com.simiacryptus.mindseye.test.unit.CudaLayerTester;
import com.simiacryptus.mindseye.test.unit.PerformanceTester;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.ArrayList;

/**
 * The type Cudnn layer apply base.
 */
public abstract class CudaLayerTestBase extends LayerTestBase {
  
  /**
   * Instantiates a new Cudnn layer apply base.
   */
  public CudaLayerTestBase() {
  }
  
  @Nonnull
  @Override
  public ArrayList<ComponentTest<?>> getBigTests() {
    @Nonnull ArrayList<ComponentTest<?>> copy = new ArrayList<>(super.getBigTests());
    if (CudaSystem.isEnabled()) copy.add(new CudaLayerTester(tolerance));
    return copy;
  }
  
  @Override
  public void run(NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    log.p(log.file((String) null, logName, "GPU Log"));
    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    CudaSystem.addLog(apiLog);
    super.run(log);
    log.p("CudaSystem Statistics:");
    log.code(() -> {
      return TestUtil.toFormattedJson(CudaSystem.getExecutionStatistics());
    });
    apiLog.close();
    CudaSystem.apiLog.remove(apiLog);
  }
  
  @Nullable
  @Override
  protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
    @Nullable final ComponentTest<ToleranceStatistics> inner = super.getReferenceIOTester();
    return new ComponentTestBase<ToleranceStatistics>() {
      @Override
      protected void _free() {
        inner.freeRef();
        super._free();
      }
      
      @Override
      public ToleranceStatistics test(@Nonnull NotebookOutput log, Layer component, Tensor... inputPrototype) {
        @Nullable PrintStream apiLog = null;
        try {
          @Nonnull String logName = "cuda_" + log.getName() + "_io.log";
          log.p(log.file((String) null, logName, "GPU Log"));
          apiLog = new PrintStream(log.file(logName));
          CudaSystem.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          if (null != apiLog) {
            apiLog.close();
            CudaSystem.apiLog.remove(apiLog);
          }
        }
      }
    };
  }
  
  
  @Nullable
  @Override
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    @Nullable ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setBatches(testingBatchSize);
    return new ComponentTestBase<ToleranceStatistics>() {
      @Override
      protected void _free() {
        inner.freeRef();
        super._free();
      }
      
      @Override
      public ToleranceStatistics test(@Nonnull NotebookOutput log, Layer component, Tensor... inputPrototype) {
        @Nullable PrintStream apiLog = null;
        try {
          @Nonnull String logName = "cuda_" + log.getName() + "_perf.log";
          log.p(log.file((String) null, logName, "GPU Log"));
          apiLog = new PrintStream(log.file(logName));
          CudaSystem.addLog(apiLog);
          return inner.test(log, component, inputPrototype);
        } finally {
          if (null != apiLog) {
            apiLog.close();
            CudaSystem.apiLog.remove(apiLog);
          }
        }
      }
    };
  }
}
