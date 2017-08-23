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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;

public final class GpuController {
  
  public static final GpuController INSTANCE = new GpuController();
  
  protected boolean verbose = true;
  Map<String, Double> deviceWeight = new HashMap<>();
  Map<String, Integer> deviceBatchSizes = new HashMap<>();LoadingCache<CuDNN, ExecutorService> gpuDriverThreads = CacheBuilder.newBuilder().build(new CacheLoader<CuDNN, ExecutorService>() {
    @Override
    public ExecutorService load(CuDNN gpu) throws Exception {
      return Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r);
        thread.setName(gpu.toString());
        return thread;
      });
    }
  });
  
  public static boolean isOom(Throwable t) {
    if(t instanceof OutOfMemoryError) return true;
    if(t instanceof DeviceOutOfMemoryError) return true;
    if(null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  public <T> T distribute(List<Tensor[]> sampledData, BiFunction<List<Tensor[]>, CudaExecutionContext, T> function, BinaryOperator<T> reducer) {
    List<CudaExecutionContext> devices = CudaExecutionContext.gpuContexts.getAll();
    double weightSum = devices.stream().mapToDouble(d -> deviceWeight.getOrDefault(d.toString(), 1.0)).sum();
    List<Future<T>> results = new ArrayList<>();
    int start = 0;
    for(int i=0;i<devices.size();i++) {
      CudaExecutionContext dev = devices.get(i);
      int sampleSize = (int) ((sampledData.size() / weightSum) * deviceWeight.getOrDefault(dev.toString(), 1.0));
      int end = start + sampleSize;
      List<Tensor[]> subList = sampledData.subList(start, end);
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
    }).reduce(reducer).get();
  }
  
  private <T> T evaluate(CudaExecutionContext gpu, List<Tensor[]> data, BiFunction<List<Tensor[]>, CudaExecutionContext, T> function, BinaryOperator<T> reducer) {
    Integer batchSize = deviceBatchSizes.getOrDefault(gpu.toString(), data.size());
    try {
      long startNanos = System.nanoTime();
      List<List<Tensor[]>> batches = Lists.partition(data, batchSize);
      T deviceResult = batches.stream().map(x-> function.apply(x, gpu)).reduce(reducer).get();
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
  
  public void cleanMemory() {
    Tensor.clear();
    System.gc();
    System.runFinalization();
  }
}
