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
import com.simiacryptus.mindseye.layers.cudnn.lang.Precision;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import org.junit.Assert;
import org.junit.Test;

import java.io.PrintStream;
import java.util.ArrayList;

/**
 * The type Cudnn layer apply base.
 */
public class CudnnTest extends NotebookReportBase {
  
  /**
   * Instantiates a new Cudnn layer apply base.
   */
  public CudnnTest() {
  }
  
  /**
   * Test.
   */
  @Test
  public void allocationOverflow() {
    run(this::allocationOverflow, "allocationOverflow");
  }
  
  private void allocationOverflow(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    CuDNN.forEach(ctx -> {
      log.h1("Device " + ctx.getDeviceNumber() + ": " + CuDNN.getDeviceName(ctx.getDeviceNumber()));
      try {
        log.code(() -> {
          ArrayList<Object> list = new ArrayList<>();
          int maxValue = Integer.MAX_VALUE - 0xFFF;
          int size = 8;
          while (true) {
            int s = size;
            TimedResult<CudaPtr> timedResult = TimedResult.time(() -> {
              return CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, new TensorArray(new Tensor(s)));
            });
            logger.info(String.format("Allocated %d in %.4fsec", size, timedResult.seconds()));
            list.add(timedResult.result);
            size = size + size / 4;
            size = size < 0 ? maxValue : Math.min(size, maxValue);
          }
        });
      } catch (Exception e) {
        CuDNN.cleanMemory();
      }
    });
    CuDNN.removeLog(apiLog);
  }
  
  @Test
  public void memoryTransfer() {
    run(this::memoryTransfer, "memoryTransfer");
  }
  
  private void memoryTransfer(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int _size = 8;
    for (int i = 0; i < 50; i++)
      try {
        int size = _size;
        Tensor original = new Tensor(size).map(this::random);
        log.code(() -> {
          CudaPtr write = CuDNN.run(ctx -> {
            TimedResult<CudaPtr> timedResult = TimedResult.time(() -> CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, new TensorArray(original)));
            logger.info(String.format("Wrote %d bytes in %.4f seconds, Device %d: %s", size, timedResult.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
            return timedResult.result;
          });
          CuDNN.forEach(ctx -> {
            Tensor readCopy = new Tensor(size);
            TimedResult<CudaPtr> timedResult = TimedResult.time(() -> write.read(Precision.Double, readCopy.getData()));
            TimedResult<Boolean> timedVerify = TimedResult.time(() -> original.equals(readCopy));
            logger.info(String.format("Read %d bytes in %.4f seconds and verified in %.4fs using device %d: %s",
                                      size, timedResult.seconds(), timedVerify.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
            Assert.assertTrue(original.prettyPrint() + " != " + readCopy.prettyPrint(), timedVerify.result);
          });
        });
        _size = _size + _size / 1;
        if (_size < 0) break;
      } catch (Exception e) {
        CuDNN.cleanMemory();
      }
    CuDNN.removeLog(apiLog);
  }
  
  private double random(double v) {
    return FastRandom.random();
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
