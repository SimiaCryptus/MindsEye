/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.GpuError;
import com.simiacryptus.mindseye.lang.TensorMemory;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;

/**
 * The type Gpu controller.
 */
public final class GpuController {
  
  /**
   * The constant INSTANCE.
   */
  public static final GpuController INSTANCE = new GpuController();
  
  /**
   * The Device weight.
   */
  Map<String, Double> deviceWeight = new HashMap<>();
  /**
   * The Device batch sizes.
   */
  Map<String, Integer> deviceBatchSizes = new HashMap<>();
  private LoadingCache<CudaExecutionContext, ExecutorService> gpuDriverThreads = CacheBuilder.newBuilder().build(new CacheLoader<CuDNN, ExecutorService>() {
    @Override
    public ExecutorService load(CuDNN gpu) throws Exception {
      return Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r);
        thread.setName(gpu.toString());
        return thread;
      });
    }
  });
  
  /**
   * Is oom boolean.
   *
   * @param t the t
   * @return the boolean
   */
  public static boolean isOom(Throwable t) {
    if (t instanceof java.lang.OutOfMemoryError) return true;
    if (t instanceof com.simiacryptus.mindseye.lang.OutOfMemoryError) return true;
    //if (t instanceof com.simiacryptus.mindseye.lang.GpuError) return true;
    if (null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
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
  public <T, U> T distribute(List<U> data, BiFunction<List<U>, CudaExecutionContext, T> mapper, BinaryOperator<T> reducer) {
    if (data.isEmpty()) return null;
    List<CudaExecutionContext> devices = CudaExecutionContext.gpuContexts.getAll();
    double weightSum = devices.stream().mapToDouble(d -> getDeviceWeight(d)).sum();
    List<Future<T>> results = new ArrayList<>();
    int start = 0;
    assert !devices.isEmpty();
    for (int i = 0; i < devices.size(); i++) {
      CudaExecutionContext dev = devices.get(i);
      int sampleSize = (int) Math.max(1, ((data.size() / weightSum) * getDeviceWeight(dev)));
      int end = start + sampleSize;
      if (i == devices.size() - 1) end = data.size();
      List<U> subList = data.subList(start, Math.min(end, data.size()));
      if (subList.isEmpty()) continue;
      try {
        results.add(getGpuDriverThreads().get(dev).submit(() -> evaluate(dev, subList, mapper, reducer)));
      } catch (ExecutionException e) {
        throw new GpuError(e);
      }
      start = end;
    }
    assert !results.isEmpty();
    return results.stream().map(x -> {
      try {
        T t = x.get();
        assert (null != t);
        return t;
      } catch (InterruptedException e) {
        throw new GpuError(e);
      } catch (ExecutionException e) {
        throw new GpuError(e);
      }
    }).reduce(reducer).orElse(null);
  }
  
  /**
   * Gets device weight.
   *
   * @param d the d
   * @return the device weight
   */
  public double getDeviceWeight(CudaExecutionContext d) {
    return 1.0;
    //return deviceWeight.getOrDefault(d.toString(), 1.0);
  }
  
  private <T, U> T evaluate(CudaExecutionContext gpu, List<U> data, BiFunction<List<U>, CudaExecutionContext, T> mapper, BinaryOperator<T> reducer) {
    Integer batchSize = deviceBatchSizes.getOrDefault(gpu.toString(), data.size());
    try {
      long startNanos = System.nanoTime();
      List<List<U>> batches = (data.size() > batchSize) ? Lists.partition(data, batchSize) : Arrays.asList(data);
      T deviceResult = batches.stream().map(x -> mapper.apply(x, gpu)).filter(x -> null != x).reduce(reducer).get();
      double time = (System.nanoTime() - startNanos) * 1.0 / 1e9;
      deviceWeight.put(gpu.toString(), data.size() / time);
      return deviceResult;
    } catch (Throwable t) {
      if (GpuController.isOom(t) && batchSize > 1) {
        batchSize = batchSize / 2;
        deviceBatchSizes.put(gpu.toString(), batchSize);
        cleanMemory();
        return evaluate(gpu, data, mapper, reducer);
      }
      else {
        RuntimeException runtimeException = new GpuError(String.format("Failed executing %s items", batchSize), t);
        runtimeException.fillInStackTrace();
        runtimeException.printStackTrace(System.err);
        throw runtimeException;
      }
    }
  }
  
  /**
   * Clean memory.
   */
  public void cleanMemory() {
    TensorMemory.clear();
    System.gc();
    System.runFinalization();
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
  public void setGpuDriverThreads(LoadingCache<CudaExecutionContext, ExecutorService> gpuDriverThreads) {
    this.gpuDriverThreads = gpuDriverThreads;
  }
}
