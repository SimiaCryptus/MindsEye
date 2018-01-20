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

package com.simiacryptus.mindseye.lang.cudnn;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.TimedResult;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Cudnn layer apply base.
 */
public class CudnnTest extends NotebookReportBase {
  
  /**
   * Test.
   */
  @Test
  @Ignore
  public void allocationOverflow() {
    run(this::allocationOverflow, "allocationOverflow");
  }
  
  private void allocationOverflow(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    GpuHandle.forEach(ctx -> {
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
        e.printStackTrace(System.out);
      }
    });
    CuDNN.removeLog(apiLog);
  }
  
  /**
   * Memory transfer.
   */
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
    for (int i = 0; i < 50; i++) {
      int size = _size;
      memoryTransfer(log, size);
      _size = _size + _size / 1;
      if (_size < 0) break;
      if (size > 512 * 1024 * 1204) break;
    }
    CuDNN.removeLog(apiLog);
  }
  
  private void memoryTransfer(NotebookOutput log, int... size) {
    Supplier<TensorList> factory = () -> new TensorArray(IntStream.range(0, 1).mapToObj(j -> {
      Tensor tensor = new Tensor(size);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    TensorList original = factory.get();
    log.code(() -> {
      CudaPtr write = GpuHandle.run(ctx -> {
        TimedResult<CudaPtr> timedResult = TimedResult.time(() -> CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, original));
        logger.info(String.format("Wrote %d bytes in %.4f seconds, Device %d: %s", size, timedResult.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        return timedResult.result;
      });
      GpuHandle.forEach(ctx -> {
        Tensor readCopy = new Tensor(size);
        TimedResult<CudaPtr> timedResult = TimedResult.time(() -> write.read(Precision.Double, readCopy.getData()));
        TimedResult<Boolean> timedVerify = TimedResult.time(() -> original.get(0).equals(readCopy));
        logger.info(String.format("Read %d bytes in %.4f seconds and verified in %.4fs using device %d: %s",
                                  size, timedResult.seconds(), timedVerify.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        if (!timedVerify.result)
          Assert.assertTrue(original.prettyPrint() + " != " + readCopy.prettyPrint(), timedVerify.result);
      });
    });
  }
  
  /**
   * Tensor lists.
   */
  @Test
  public void tensorLists() {
    run(this::tensorLists, "tensorLists");
  }
  
  private void tensorLists(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int size = 8;
    for (int i = 0; i < 50; i++) {
      int length = 10;
      int accumulations = 3;
      double memoryLoadCoeff = length * accumulations / 3;
      testTensorList(log, new int[]{size}, length, 1e-2, accumulations);
      size = size + size / 1;
      if (size < 0) break;
      if (size > memoryLoadCoeff * 256 * 1024 * 1204) break;
    }
    CuDNN.removeLog(apiLog);
  }
  
  private void testTensorList(NotebookOutput log, int[] dimensions, int length, double tolerance, int accumulations) {
    Supplier<TensorList> factory = () -> new TensorArray(IntStream.range(0, length).mapToObj(j -> {
      Tensor tensor = new Tensor(dimensions);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    log.code(() -> {
      TimedResult<TensorList> originalTiming = TimedResult.time(() -> factory.get());
      logger.info(String.format("Calculated test data in %.4fsec", originalTiming.seconds()));
      TensorList original = originalTiming.result;
      TensorList mutableGpuData = GpuHandle.run(ctx -> {
        TimedResult<CudaPtr> timedResult = TimedResult.time(() -> CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, original));
        logger.info(String.format("Wrote %s in %.4f seconds, Device %d: %s", Arrays.toString(dimensions), timedResult.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        return GpuTensorList.create(timedResult.result, length, dimensions, Precision.Double);
      });
      GpuHandle.forEach(ctx -> {
        TimedResult<TensorList> timedResult = TimedResult.time(() -> (mutableGpuData instanceof GpuTensorList) ? ((GpuTensorList) mutableGpuData).getHeapCopy() : mutableGpuData);
        TimedResult<Boolean> timedVerify = TimedResult.time(() -> original.minus(timedResult.result).stream().flatMapToDouble(x -> Arrays.stream(x.getData())).map(x -> Math.abs(x)).max().getAsDouble() < tolerance);
        logger.info(String.format("Read %s in %.4f seconds and verified in %.4fs using device %d: %s",
                                  Arrays.toString(dimensions), timedResult.seconds(), timedVerify.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        if (!timedVerify.result)
          Assert.assertTrue(original.prettyPrint() + " != " + timedResult.result.prettyPrint(), timedVerify.result);
      });
      TimedResult<List<TensorList>> accumulantTiming = TimedResult.time(() -> IntStream.range(0, accumulations).mapToObj(x -> factory.get()).collect(Collectors.toList()));
      logger.info(String.format("Calculated accumulant in %.4fsec", accumulantTiming.seconds()));
      List<TensorList> accumulants = accumulantTiming.result;
      accumulants.stream().forEach(accumulant -> {
        GpuHandle.apply(ctx -> {
          TimedResult<TensorList> timedWrite = TimedResult.time(() -> {
            return GpuTensorList.create(CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, accumulant), length, dimensions, Precision.Double);
          });
          TimedResult<Void> timedAccumulation = TimedResult.time(() -> mutableGpuData.addInPlace(timedWrite.result));
          logger.info(String.format("Wrote in %.4f seconds and accumulated %s in %.4f seconds, Device %d: %s",
                                    timedAccumulation.seconds(), Arrays.toString(dimensions), timedWrite.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        });
      });
      TimedResult<TensorList> finalResultTiming = TimedResult.time(() -> {
        Stream<TensorList> stream = Stream.generate(() -> accumulants.remove(0)).limit(accumulants.size());
        return stream.reduce((a, b) -> a.add(b)).map(x -> x.add(original)).orElse(original);
      });
      logger.info(String.format("Calculated final data in %.4fsec", finalResultTiming.seconds()));
      TensorList finalResult = finalResultTiming.result;
      GpuHandle.forEach(ctx -> {
        TimedResult<Boolean> timedVerify = TimedResult.time(() -> finalResult.minus(mutableGpuData).stream().flatMapToDouble(x -> Arrays.stream(x.getData())).map(x -> Math.abs(x)).max().getAsDouble() < tolerance);
        logger.info(String.format("Read %s and verified in %.4fs using device %d: %s",
                                  Arrays.toString(dimensions), timedVerify.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
        if (!timedVerify.result)
          Assert.assertTrue(finalResult.prettyPrint() + " != " + mutableGpuData.prettyPrint(), timedVerify.result);
      });
    });
  }
  
  /**
   * Tensor lists multithreaded.
   */
  @Test
  public void tensorLists_multithreaded() {
    run(this::tensorLists_multithreaded, "tensorLists_multithreaded");
  }
  
  private void tensorLists_multithreaded(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + ".log";
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int size = 8;
    for (int i = 0; i < 50; i++) {
      int length = 10;
      int accumulations = 5;
      double memoryLoadCoeff = length * accumulations / 3;
      testTensorListMT(log, new int[]{size}, length, 1e-2, accumulations);
      size = size + size / 1;
      if (size < 0) break;
      if (size > memoryLoadCoeff * 128 * 1024 * 1204) break;
    }
    CuDNN.removeLog(apiLog);
  }
  
  private void testTensorListMT(NotebookOutput log, int[] dimensions, int length, double tolerance, int accumulations) {
    Supplier<TensorList> factory = () -> new TensorArray(IntStream.range(0, length).mapToObj(j -> {
      Tensor tensor = new Tensor(dimensions);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    log.code(() -> {
      ListeningExecutorService pool = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(5));
      PrintStream out = SysOutInterceptor.INSTANCE.currentHandler();
      try {
        Futures.allAsList(IntStream.range(0, 5).mapToObj(workerNumber -> {
          TimedResult<TensorList> originalTiming = TimedResult.time(() -> factory.get());
          TensorList original = originalTiming.result;
          logger.info(String.format("[%s] Calculated test data in %.4fsec", workerNumber, originalTiming.seconds()));
          ListenableFuture<TensorList> mutableDataFuture = pool.submit(() -> GpuHandle.run(ctx -> {
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            TimedResult<CudaPtr> timedResult = TimedResult.time(() -> CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, original));
            logger.info(String.format("[%s] Wrote %s in %.4f seconds, Device %d: %s", workerNumber, Arrays.toString(dimensions), timedResult.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            return GpuTensorList.create(timedResult.result, length, dimensions, Precision.Double);
          }));
          TimedResult<List<TensorList>> accumulantTiming = TimedResult.time(() -> IntStream.range(0, accumulations).mapToObj(x -> factory.get()).collect(Collectors.toList()));
          List<TensorList> accumulants = accumulantTiming.result;
          logger.info(String.format("[%s] Calculated accumulant in %.4fsec", workerNumber, accumulantTiming.seconds()));
          ListenableFuture<TensorList> accumulated = Futures.transform(mutableDataFuture, (mutableGpuData) -> {
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            accumulants.stream().forEach(delta -> {
              GpuHandle.apply(ctx -> {
                TimedResult<GpuTensorList> timedWrite = TimedResult.time(() -> {
                  return GpuTensorList.create(CudaPtr.write(ctx.getDeviceNumber(), Precision.Double, delta), length, dimensions, Precision.Double);
                });
                TimedResult<Void> timedAccumulation = TimedResult.time(() -> mutableGpuData.addInPlace(timedWrite.result));
                logger.info(String.format("[%s] Wrote in %.4f seconds and accumulated %s in %.4f seconds, Device %d: %s", workerNumber,
                                          timedAccumulation.seconds(), Arrays.toString(dimensions), timedWrite.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
              });
            });
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            return mutableGpuData;
          }, pool);
          TimedResult<TensorList> finalResultTiming = TimedResult.time(() -> {
            return accumulants.stream().reduce((a, b) -> a.add(b)).map(x -> x.add(original)).orElse(original);
          });
          TensorList finalResult = finalResultTiming.result;
          logger.info(String.format("[%s] Calculated final data in %.4fsec", workerNumber, finalResultTiming.seconds()));
          return Futures.transform(accumulated, (write) -> {
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            GpuHandle.apply(ctx -> {
              TimedResult<Boolean> timedVerify = TimedResult.time(() -> finalResult.minus(write).stream().flatMapToDouble(x -> Arrays.stream(x.getData())).map(x -> Math.abs(x)).max().getAsDouble() < tolerance);
              logger.info(String.format("[%s] Read %s and verified in %.4fs using device %d: %s", workerNumber,
                                        Arrays.toString(dimensions), timedVerify.seconds(), ctx.getDeviceNumber(), CuDNN.getDeviceName(ctx.getDeviceNumber())));
              if (!timedVerify.result)
                Assert.assertTrue(finalResult.prettyPrint() + " != " + write.prettyPrint(), timedVerify.result);
            });
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            return null;
          }, pool);
        }).collect(Collectors.toList())).get();
      } catch (InterruptedException | ExecutionException e) {
        throw new RuntimeException(e);
      } finally {
        pool.shutdown();
      }
    });
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
