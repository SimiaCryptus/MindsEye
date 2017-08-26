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
import com.simiacryptus.util.ml.Tensor;

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
   * The Verbose.
   */
  protected boolean verbose = false;
  /**
   * The Device weight.
   */
  Map<String, Double> deviceWeight = new HashMap<>();
  /**
   * The Device batch sizes.
   */
  Map<String, Integer> deviceBatchSizes = new HashMap<>();
  /**
   * The Gpu driver threads.
   */
  LoadingCache<CuDNN, ExecutorService> gpuDriverThreads = CacheBuilder.newBuilder().build(new CacheLoader<CuDNN, ExecutorService>() {
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
    if(t instanceof OutOfMemoryError) return true;
    if(t instanceof DeviceOutOfMemoryError) return true;
    if(null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  /**
   * Distribute t.
   *
   * @param <T>         the type parameter
   * @param sampledData the sampled data
   * @param function    the function
   * @param reducer     the reducer
   * @return the t
   */
  public <T> T distribute(List<Tensor[]> sampledData, BiFunction<List<Tensor[]>, CudaExecutionContext, T> function, BinaryOperator<T> reducer) {
    if(sampledData.isEmpty()) return null;
    List<CudaExecutionContext> devices = CudaExecutionContext.gpuContexts.getAll();
    double weightSum = devices.stream().mapToDouble(d -> deviceWeight.getOrDefault(d.toString(), 1.0)).sum();
    List<Future<T>> results = new ArrayList<>();
    int start = 0;
    for(int i=0;i<devices.size();i++) {
      CudaExecutionContext dev = devices.get(i);
      int sampleSize = (int) Math.max(1, ((sampledData.size() / weightSum) * deviceWeight.getOrDefault(dev.toString(), 1.0)));
      int end = start + sampleSize;
      List<Tensor[]> subList = sampledData.subList(start, Math.min(end, sampledData.size()));
      if(subList.isEmpty()) continue;
      try {
        results.add(gpuDriverThreads.get(dev).submit(() -> evaluate(dev, subList, function, reducer)));
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
      start = end;
    }
    return results.stream().map(x -> {
      try {
        return x.get();
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
    }).reduce(reducer).orElse(null);
  }
  
  private <T> T evaluate(CudaExecutionContext gpu, List<Tensor[]> data, BiFunction<List<Tensor[]>, CudaExecutionContext, T> function, BinaryOperator<T> reducer) {
    Integer batchSize = deviceBatchSizes.getOrDefault(gpu.toString(), data.size());
    try {
      long startNanos = System.nanoTime();
      List<List<Tensor[]>> batches = (data.size() > batchSize) ? Lists.partition(data, batchSize) : Arrays.asList(data);
      T deviceResult = batches.stream().map(x-> function.apply(x, gpu)).filter(x->null != x).reduce(reducer).get();
      double time = (System.nanoTime() - startNanos) * 1.0 / 1e9;
      if(verbose) System.out.println(String.format("Device %s completed %s items in %s sec", gpu, data.size(), time));
      deviceWeight.put(gpu.toString(), data.size() / time);
      return deviceResult;
    } catch (Throwable t) {
      if(GpuController.isOom(t) && batchSize > 1) {
        batchSize = batchSize / 2;
        deviceBatchSizes.put(gpu.toString(), batchSize);
        cleanMemory();
        return evaluate(gpu, data, function, reducer);
      } else {
        RuntimeException runtimeException = new RuntimeException(String.format("Failed executing %s items", batchSize), t);
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
    Tensor.clear();
    System.gc();
    System.runFinalization();
  }
}
