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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Cudnn layer run base.
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
  
  private void allocationOverflow(@Nonnull NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    CudaSystem.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    CudnnHandle.forEach(gpu -> {
      log.h1("Device " + gpu.getDeviceId() + ": " + CudaDevice.getDeviceName(gpu.getDeviceId()));
      try {
        log.code(() -> {
          @Nonnull ArrayList<Object> list = new ArrayList<>();
          int maxValue = Integer.MAX_VALUE - 0xFFF;
          int size = 8;
          while (true) {
            int s = size;
            @Nonnull TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
              return gpu.getTensor(TensorArray.create(new Tensor(s)), Precision.Double, MemoryType.Managed, false);
            });
            logger.info(String.format("Allocated %d in %.4fsec", size, timedResult.seconds()));
            list.add(timedResult.result);
            size = size + size / 4;
            size = size < 0 ? maxValue : Math.min(size, maxValue);
          }
        });
      } catch (Exception e) {
        logger.warn("Error allocating", e);
        System.gc();
      }
    });
    CudaSystem.removeLog(apiLog);
  }
  
  /**
   * Memory transfer.
   */
  @Test
  public void memoryTransfer() {
    run(this::memoryTransfer, "memoryTransfer");
  }
  
  private void memoryTransfer(@Nonnull NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    CudaSystem.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int _size = 8;
    for (int i = 0; i < 10; i++) {
      int size = _size;
      memoryTransfer(log, size);
      _size = _size + _size / 1;
      if (_size < 0) break;
      if (size > 512 * 1024 * 1204) break;
    }
    CudaSystem.removeLog(apiLog);
  }
  
  private void memoryTransfer(@Nonnull NotebookOutput log, int... size) {
    @Nonnull Supplier<TensorList> factory = () -> TensorArray.wrap(IntStream.range(0, 1).mapToObj(j -> {
      @Nonnull Tensor tensor = new Tensor(size);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    TensorList original = factory.get();
    log.code(() -> {
      CudaTensor write = CudaSystem.eval(gpu -> {
        @Nonnull TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
          return gpu.getTensor(original, Precision.Double, MemoryType.Managed, false);
        });
        int deviceNumber = gpu.getDeviceId();
        logger.info(String.format("Wrote %s bytes in %.4f seconds, Device %d: %s", Arrays.toString(size), timedResult.seconds(), deviceNumber, CudaDevice.getDeviceName(deviceNumber)));
        return timedResult.result;
      }, original);
      CudnnHandle.forEach(gpu -> {
        @Nonnull Tensor readCopy = new Tensor(size);
        @Nonnull TimedResult<CudaMemory> timedResult = TimedResult.time(() -> {
          CudaMemory cudaMemory = write.getMemory(gpu);
          CudaMemory read = cudaMemory.read(Precision.Double, readCopy.getData());
          cudaMemory.freeRef();
          return read;
        });
        @Nonnull TimedResult<Boolean> timedVerify = TimedResult.time(() -> {
          @Nullable Tensor tensor = original.get(0);
          boolean equals = tensor.equals(readCopy);
          tensor.freeRef();
          return equals;
        });
        int deviceNumber = gpu.getDeviceId();
        logger.info(String.format("Read %s bytes in %.4f seconds and verified in %.4fs using device %d: %s",
          Arrays.toString(size), timedResult.seconds(), timedVerify.seconds(), deviceNumber, CudaDevice.getDeviceName(deviceNumber)));
        if (!timedVerify.result)
          Assert.assertTrue(original.prettyPrint() + " != " + readCopy.prettyPrint(), timedVerify.result);
        readCopy.freeRef();
      });
      write.freeRef();
    });
    original.freeRef();
  }
  
  /**
   * Tensor lists.
   */
  @Test
  public void tensorLists() {
    run(this::tensorLists, "tensorLists");
  }
  
  private void tensorLists(@Nonnull NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    CudaSystem.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int size = 8;
    for (int i = 0; i < 18; i++) {
      log.out("Loop " + i);
      int length = 10;
      int accumulations = 3;
      double memoryLoadCoeff = length * accumulations / 3;
      testTensorList(log, new int[]{size}, length, 1e-2, accumulations);
      size = size + size / 1;
      if (size < 0) break;
      if (size > memoryLoadCoeff * 256 * 1024 * 1204) break;
    }
    CudaSystem.removeLog(apiLog);
  }
  
  private void testTensorList(@Nonnull NotebookOutput log, @Nonnull int[] dimensions, int length, double tolerance, int accumulations) {
    @Nonnull Supplier<TensorList> factory = () -> TensorArray.wrap(IntStream.range(0, length).mapToObj(j -> {
      @Nonnull Tensor tensor = new Tensor(dimensions);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    log.code(() -> {
      @Nonnull TimedResult<TensorList> originalTiming = TimedResult.time(() -> factory.get());
      logger.info(String.format("Calculated test data in %.4fsec", originalTiming.seconds()));
      TensorList original = originalTiming.result;
      @Nonnull AtomicReference<TensorList> mutableGpuData = new AtomicReference<>(CudaSystem.eval(gpu -> {
        @Nonnull TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
          return gpu.getTensor(original, Precision.Double, MemoryType.Managed, false);
        });
        logger.info(String.format("Wrote %s in %.4f seconds, Device %d: %s", Arrays.toString(dimensions), timedResult.seconds(), gpu.getDeviceId(), CudaDevice.getDeviceName(gpu.getDeviceId())));
        return CudaTensorList.wrap(timedResult.result, length, dimensions, Precision.Double);
      }, original));
      CudnnHandle.forEach(ctx -> {
        @Nonnull TimedResult<TensorList> timedResult = TimedResult.time(() -> (mutableGpuData.get() instanceof CudaTensorList) ? ((CudaTensorList) mutableGpuData.get()).getHeapCopy() : mutableGpuData.get());
        @Nonnull TimedResult<Boolean> timedVerify = TimedResult.time(() -> {
          @Nonnull TensorList minus = original.minus(timedResult.result);
          double variance = minus.stream().mapToDouble(x -> Arrays.stream(x.getData()).map(Math::abs).max().getAsDouble())
            .max().getAsDouble();
          minus.freeRef();
          return variance < tolerance;
        });
        logger.info(String.format("Read %s in %.4f seconds and verified in %.4fs using device %d: %s",
          Arrays.toString(dimensions), timedResult.seconds(), timedVerify.seconds(), ctx.getDeviceId(), CudaDevice.getDeviceName(ctx.getDeviceId())));
        if (!timedVerify.result)
          Assert.assertTrue(original.prettyPrint() + " != " + timedResult.result.prettyPrint(), timedVerify.result);
        timedResult.result.freeRef();
      });
      @Nonnull TimedResult<List<TensorList>> accumulantTiming = TimedResult.time(() -> IntStream.range(0, accumulations).mapToObj(x -> factory.get()).collect(Collectors.toList()));
      logger.info(String.format("Calculated accumulant in %.4fsec", accumulantTiming.seconds()));
      List<TensorList> accumulants = accumulantTiming.result;
      accumulants.stream().forEach(accumulant -> {
        CudaSystem.run(gpu -> {
          @Nonnull TimedResult<TensorList> timedWrite = TimedResult.time(() -> {
            return CudaTensorList.wrap(gpu.getTensor(accumulant, Precision.Double, MemoryType.Managed, false), length, dimensions, Precision.Double);
          });
          @Nonnull TimedResult<Void> timedAccumulation = TimedResult.time(() -> {
            mutableGpuData.getAndUpdate(x -> x.add(timedWrite.result)).freeRef();
            timedWrite.result.freeRef();
          });
          logger.info(String.format("Wrote in %.4f seconds and accumulated %s in %.4f seconds, Device %d: %s",
            timedAccumulation.seconds(), Arrays.toString(dimensions), timedWrite.seconds(), gpu.getDeviceId(), CudaDevice.getDeviceName(gpu.getDeviceId())));
        }, accumulant);
      });
      @Nonnull TimedResult<TensorList> finalResultTiming = TimedResult.time(() -> {
        return accumulants.stream().reduce((a, b) -> {
          TensorList sum = a.addAndFree(b);
          b.freeRef();
          return sum;
        }).map(x -> {
          TensorList sum = x.add(original);
          x.freeRef();
          return sum;
        }).orElseGet(() -> {
          original.addRef();
          return original;
        });
      });
      original.freeRef();
      logger.info(String.format("Calculated final data in %.4fsec", finalResultTiming.seconds()));
      TensorList finalResult = finalResultTiming.result;
      CudnnHandle.forEach(ctx -> {
        @Nonnull TimedResult<Boolean> timedVerify = TimedResult.time(() -> {
          @Nonnull TensorList minus = finalResult.minus(mutableGpuData.get());
          double diffVal = minus.stream().mapToDouble(x -> {
            double v = Arrays.stream(x.getData()).map(Math::abs).max().getAsDouble();
            x.freeRef();
            return v;
          }).max().getAsDouble();
          minus.freeRef();
          return diffVal < tolerance;
        });
        logger.info(String.format("Read %s and verified in %.4fs using device %d: %s",
          Arrays.toString(dimensions), timedVerify.seconds(), ctx.getDeviceId(), CudaDevice.getDeviceName(ctx.getDeviceId())));
        if (!timedVerify.result)
          Assert.assertTrue(finalResult.prettyPrint() + " != " + mutableGpuData.get().prettyPrint(), timedVerify.result);
      });
      mutableGpuData.get().freeRef();
      finalResult.freeRef();
    });
  }
  
  /**
   * Tensor lists multithreaded.
   */
  @Test
  public void tensorLists_multithreaded() {
    run(this::tensorLists_multithreaded, "tensorLists_multithreaded");
  }
  
  private void tensorLists_multithreaded(@Nonnull NotebookOutput log) {
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    CudaSystem.addLog(apiLog);
    log.p(log.file((String) null, logName, "GPU Log"));
    int size = 8;
    for (int i = 0; i < 12; i++) {
      log.out("Loop " + i);
      int length = 10;
      int accumulations = 100;
      double memoryLoadCoeff = length * accumulations / 3;
      testTensorListMT(log, new int[]{size}, length, 1e-2, accumulations);
      size = size + size / 1;
      if (size < 0) break;
      if (size > memoryLoadCoeff * 128 * 1024 * 1204) break;
    }
    CudaSystem.removeLog(apiLog);
  }
  
  private void testTensorListMT(@Nonnull NotebookOutput log, @Nonnull int[] dimensions, int length, double tolerance, int accumulations) {
    @Nonnull Supplier<TensorList> factory = () -> TensorArray.wrap(IntStream.range(0, length).mapToObj(j -> {
      @Nonnull Tensor tensor = new Tensor(dimensions);
      Arrays.parallelSetAll(tensor.getData(), this::random);
      return tensor;
    }).toArray(j -> new Tensor[j]));
    log.code(() -> {
      @Nonnull ListeningExecutorService pool = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(5));
      PrintStream out = SysOutInterceptor.INSTANCE.currentHandler();
      try {
        List<ListenableFuture<Object>> collect = IntStream.range(0, 16).mapToObj(workerNumber -> {
          @Nonnull TimedResult<TensorList> originalTiming = TimedResult.time(() -> factory.get());
          TensorList original = originalTiming.result;
          logger.info(String.format("[%s] Calculated test data in %.4fsec", workerNumber, originalTiming.seconds()));
          @Nonnull ListenableFuture<TensorList> mutableDataFuture = pool.submit(() -> CudaSystem.eval(gpu -> {
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            @Nonnull TimedResult<CudaTensor> timedResult = TimedResult.time(() -> {
              return gpu.getTensor(original, Precision.Double, MemoryType.Managed, false);
            });
            logger.info(String.format("[%s] Wrote %s in %.4f seconds, Device %d: %s", workerNumber, Arrays.toString(dimensions), timedResult.seconds(), gpu.getDeviceId(), CudaDevice.getDeviceName(gpu.getDeviceId())));
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            return CudaTensorList.wrap(timedResult.result, length, dimensions, Precision.Double);
          }, original));
          @Nonnull TimedResult<List<TensorList>> accumulantTiming = TimedResult.time(() -> IntStream.range(0, accumulations).mapToObj(x -> factory.get()).collect(Collectors.toList()));
          List<TensorList> accumulants = accumulantTiming.result;
  
          @Nonnull TimedResult<TensorList> finalResultTiming = TimedResult.time(() -> {
            return accumulants.stream().map(x -> {
              x.addRef();
              return x;
            }).reduce((a, b) -> {
              TensorList sum = a.addAndFree(b);
              b.freeRef();
              return sum;
            }).map(x -> {
              TensorList sum = x.add(original);
              x.freeRef();
              return sum;
            }).orElseGet(() -> {
              original.addRef();
              return original;
            });
          });
          
          logger.info(String.format("[%s] Calculated accumulant in %.4fsec", workerNumber, accumulantTiming.seconds()));
          @Nonnull ListenableFuture<TensorList> accumulated = Futures.transform(mutableDataFuture, (x) -> {
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            @Nonnull AtomicReference<TensorList> mutableGpuData = new AtomicReference<>(x);
            accumulants.stream().parallel().forEach(delta -> {
              CudaSystem.run(gpu -> {
                @Nonnull TimedResult<CudaTensorList> timedWrite = TimedResult.time(() -> {
                  @Nullable CudaTensor cudaMemory = gpu.getTensor(delta, Precision.Double, MemoryType.Managed, false);
                  delta.freeRef();
                  return CudaTensorList.wrap(cudaMemory, length, dimensions, Precision.Double);
                });
                @Nonnull TimedResult<Void> timedAccumulation = TimedResult.time(() -> {
                  synchronized (mutableGpuData) {
                    mutableGpuData.getAndUpdate(y -> {
                      TensorList add = y.add(timedWrite.result);
                      y.freeRef();
                      return add;
                    });
                  }
                  timedWrite.result.freeRef();
                });
                logger.info(String.format("[%s] Wrote in %.4f seconds and accumulated %s in %.4f seconds, Device %d: %s", workerNumber,
                  timedAccumulation.seconds(), Arrays.toString(dimensions), timedWrite.seconds(), gpu.getDeviceId(), CudaDevice.getDeviceName(gpu.getDeviceId())));
              }, delta);
            });
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            return mutableGpuData.get();
          }, pool);
          TensorList finalResult = finalResultTiming.result;
          logger.info(String.format("[%s] Calculated final data in %.4fsec", workerNumber, finalResultTiming.seconds()));
          return Futures.transform(accumulated, (write) -> {
            original.freeRef();
            PrintStream oldHandler = SysOutInterceptor.INSTANCE.setCurrentHandler(out);
            CudaSystem.run(gpu -> {
              @Nonnull TimedResult<Boolean> timedVerify = TimedResult.time(() -> {
                @Nonnull TensorList minus = finalResult.minus(write);
                double diffVal = minus.stream().mapToDouble(x -> {
                  double v = Arrays.stream(x.getData()).map(Math::abs).max().getAsDouble();
                  x.freeRef();
                  return v;
                }).max().getAsDouble();
                minus.freeRef();
                return diffVal < tolerance;
              });
              logger.info(String.format("[%s] Read %s and verified in %.4fs using device %d: %s", workerNumber,
                Arrays.toString(dimensions), timedVerify.seconds(), gpu.getDeviceId(), CudaDevice.getDeviceName(gpu.getDeviceId())));
              if (!timedVerify.result)
                Assert.assertTrue(finalResult.prettyPrint() + " != " + write.prettyPrint(), timedVerify.result);
              write.freeRef();
            });
            SysOutInterceptor.INSTANCE.setCurrentHandler(oldHandler);
            finalResult.freeRef();
            return null;
          }, pool);
        }).collect(Collectors.toList());
        List<Object> objects = Futures.allAsList(collect).get();
      } catch (@Nonnull InterruptedException | ExecutionException e) {
        throw new RuntimeException(e);
      } finally {
        pool.shutdown();
      }
    });
  }
  
  private double random(double v) {
    return FastRandom.INSTANCE.random();
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }
  
  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return CudaSystem.class;
  }
}
