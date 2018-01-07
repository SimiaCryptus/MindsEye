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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.mindseye.lang.GpuError;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.util.test.SysOutInterceptor;

import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Manages the GPU resources and related execution threadpools.
 */
public final class GpuController {
  
  /**
   * The constant INSTANCE.
   */
  public static final GpuController INSTANCE = new GpuController();
  private static final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The Device batch sizes.
   */
  Map<String, Integer> deviceBatchSizes = new HashMap<>();
  /**
   * The Device weight.
   */
  Map<String, Double> deviceWeight = new HashMap<>();
  private LoadingCache<CudaExecutionContext, ExecutorService> gpuDriverThreads = CacheBuilder.newBuilder().build(new CacheLoader<CuDNN, ExecutorService>() {
    @Override
    public ExecutorService load(final CuDNN gpu) throws Exception {
      return Executors.newSingleThreadExecutor(r -> {
        final Thread thread = new Thread(r);
        thread.setName(gpu.toString());
        thread.setDaemon(true);
        return thread;
      });
    }
  });
  
  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  public static <T> T call(final Function<CudaExecutionContext, T> fn) {
    if (CudaExecutionContext.gpuContexts.getAll().isEmpty()) {
      return fn.apply(new CudaExecutionContext(-1));
    }
    else {
      return CudaExecutionContext.gpuContexts.run(exe -> {
        try {
          return GpuController.INSTANCE.getGpuDriverThreads().get(exe).submit(() -> fn.apply(exe)).get();
        } catch (final Exception e) {
          throw new RuntimeException(e);
        }
      });
    }
  }
  
  /**
   * Is oom boolean.
   *
   * @param t the t
   * @return the boolean
   */
  public static boolean isOom(final Throwable t) {
    if (t instanceof java.lang.OutOfMemoryError) return true;
    if (t instanceof com.simiacryptus.mindseye.lang.OutOfMemoryError) return true;
    //if (t instanceof com.simiacryptus.mindseye.lang.GpuError) return true;
    if (null != t.getCause() && t != t.getCause()) return GpuController.isOom(t.getCause());
    return false;
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void run(final Consumer<CudaExecutionContext> fn) {
    CudaExecutionContext.gpuContexts.apply(exe -> {
      try {
        GpuController.INSTANCE.getGpuDriverThreads().get(exe).submit(() -> fn.accept(exe)).get();
      } catch (final Exception e) {
        throw new RuntimeException(e);
      }
    });
  }
  
  /**
   * Clean memory.
   */
  public static void cleanMemory() {
    GpuController.singleThreadExecutor.submit(() -> {
      RecycleBin.DOUBLES.clear();
      System.gc();
      System.runFinalization();
    });
  }
  
  /**
   * Reset all GPUs and Heap Memory
   */
  public static void reset() {
    cleanMemory();
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    for (CudaExecutionContext exe : CudaExecutionContext.gpuContexts.getAll()) {
      try {
        GpuController.INSTANCE.getGpuDriverThreads().get(exe).submit(() -> {
          exe.initThread();
          CuDNN.cudaDeviceReset();
        }).get();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }
  
  
  /**
   * Distribute t.
   *
   * @param <T>     the type parameter
   * @param <U>     the type parameter
   * @param data    the data
   * @param mapper  the mapper
   * @param reducer the reducer
   * @return the t
   */
  public <T, U> T distribute(final List<U> data, final BiFunction<List<U>, CudaExecutionContext, T> mapper, final BinaryOperator<T> reducer) {
    if (data.isEmpty()) return null;
    final List<CudaExecutionContext> devices = CudaExecutionContext.gpuContexts.getAll();
    final double weightSum = devices.stream().mapToDouble(d -> getDeviceWeight(d)).sum();
    final List<Future<T>> results = new ArrayList<>();
    int start = 0;
    assert !devices.isEmpty();
    for (int i = 0; i < devices.size(); i++) {
      final CudaExecutionContext dev = devices.get(i);
      final int sampleSize = (int) Math.max(1, data.size() / weightSum * getDeviceWeight(dev));
      int end = start + sampleSize;
      if (i == devices.size() - 1) {
        end = data.size();
      }
      final List<U> subList = data.subList(start, Math.min(end, data.size()));
      if (subList.isEmpty()) {
        continue;
      }
      try {
        final PrintStream centralOut = SysOutInterceptor.INSTANCE.currentHandler();
        results.add(getGpuDriverThreads().get(dev).submit(() -> {
          final PrintStream defaultOut = SysOutInterceptor.INSTANCE.currentHandler();
          SysOutInterceptor.INSTANCE.setCurrentHandler(centralOut);
          final T result = evaluate(dev, subList, mapper, reducer);
          SysOutInterceptor.INSTANCE.setCurrentHandler(defaultOut);
          return result;
        }));
      } catch (final ExecutionException e) {
        throw new GpuError(e);
      }
      start = end;
    }
    assert !results.isEmpty();
    return results.stream().map(x -> {
      try {
        final T t = x.get();
        assert null != t;
        return t;
      } catch (final InterruptedException e) {
        throw new GpuError(e);
      } catch (final ExecutionException e) {
        throw new GpuError(e);
      }
    }).reduce(reducer).orElse(null);
  }
  
  private <T, U> T evaluate(final CudaExecutionContext gpu, final List<U> data, final BiFunction<List<U>, CudaExecutionContext, T> mapper, final BinaryOperator<T> reducer) {
    Integer batchSize = deviceBatchSizes.getOrDefault(gpu.toString(), data.size());
    try {
      final long startNanos = System.nanoTime();
      final List<List<U>> batches = data.size() > batchSize ? Lists.partition(data, batchSize) : Arrays.asList(data);
      final T deviceResult = batches.stream().map(x -> mapper.apply(x, gpu)).filter(x -> null != x).reduce(reducer).orElse(null);
      final double time = (System.nanoTime() - startNanos) * 1.0 / 1e9;
      deviceWeight.put(gpu.toString(), data.size() / time);
      return deviceResult;
    } catch (final Throwable t) {
      if (GpuController.isOom(t) && batchSize > 1) {
        batchSize = batchSize / 2;
        deviceBatchSizes.put(gpu.toString(), batchSize);
        cleanMemory();
        return evaluate(gpu, data, mapper, reducer);
      }
      else {
        final RuntimeException runtimeException = new GpuError(String.format("Failed executing %s items", batchSize), t);
        runtimeException.fillInStackTrace();
        runtimeException.printStackTrace(System.err);
        throw runtimeException;
      }
    }
  }
  
  /**
   * Gets device weight.
   *
   * @param d the d
   * @return the device weight
   */
  public double getDeviceWeight(final CudaExecutionContext d) {
    return 1.0;
    //return deviceWeight.getOrDefault(d.toString(), 1.0);
  }
  
  /**
   * Gets gpu driver threads.
   *
   * @return the gpu driver threads
   */
  public LoadingCache<CudaExecutionContext, ExecutorService> getGpuDriverThreads() {
    return gpuDriverThreads;
  }
  
  /**
   * Sets gpu driver threads.
   *
   * @param gpuDriverThreads the gpu driver threads
   */
  public void setGpuDriverThreads(final LoadingCache<CudaExecutionContext, ExecutorService> gpuDriverThreads) {
    this.gpuDriverThreads = gpuDriverThreads;
  }
}
