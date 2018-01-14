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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.layers.cudnn.lang.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.lang.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.lang.GpuController;
import com.simiacryptus.mindseye.layers.cudnn.lang.Precision;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import java.io.PrintStream;
import java.util.ArrayList;

/**
 * The type Cudnn layer run base.
 */
public class CudnnTest extends NotebookReportBase {
  
  /**
   * Instantiates a new Cudnn layer run base.
   */
  public CudnnTest() {
  }
  
  /**
   * Test.
   */
  @Test
  public void test() {
    run(this::test);
  }
  
  private void test(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    CuDNN.gpuContexts.getAll().forEach(ctx -> {
      log.h1("Device " + ctx.getDeviceNumber());
      try {
        log.code(() -> {
          ArrayList<Object> list = new ArrayList<>();
          CuDNN.setDevice(ctx.getDeviceNumber());
          while (true) {
            list.add(CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, new TensorArray(new Tensor(128 * 1024 * 1024))));
          }
        });
      } catch (Exception e) {
        GpuController.reset();
      }
    });
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return CuDNN.class;
  }
}
