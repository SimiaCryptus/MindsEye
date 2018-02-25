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

import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.data.DoubleStatistics;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaStream_t;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Main library wrapper class around the CudaSystem API, providing logging and managed wrappers.
 */
public class CudaSystem {
  
  /**
   * The constant INSTANCE.
   */
//  public static final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The constant apiLog.
   */
  public static final HashSet<PrintStream> apiLog = new HashSet<>();
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CudaSystem.class);
  /**
   * The constant propertyCache.
   */
  protected static final ConcurrentHashMap<Integer, cudaDeviceProp> propertyCache = new ConcurrentHashMap<>();
  /**
   * The constant currentDevice.
   */
  protected static final ThreadLocal<Integer> currentDevice = new ThreadLocal<Integer>() {
    @javax.annotation.Nonnull
    @Override
    protected Integer initialValue() {
      return -1;
    }
  };
  /**
   * The constant logThread.
   */
  protected static final ExecutorService logThread = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The constant start.
   */
  protected static final long start = System.nanoTime();
  /**
   * The constant createPoolingDescriptor_execution.
   */
  protected static final DoubleStatistics createPoolingDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceReset_execution.
   */
  protected static final DoubleStatistics cudaDeviceReset_execution = new DoubleStatistics();
  /**
   * The constant cudaFree_execution.
   */
  protected static final DoubleStatistics cudaFree_execution = new DoubleStatistics();
  /**
   * The constant cudaMalloc_execution.
   */
  protected static final DoubleStatistics cudaMalloc_execution = new DoubleStatistics();
  
  /**
   * The constant cudaDeviceSynchronize_execution.
   */
  protected static final DoubleStatistics cudaDeviceSynchronize_execution = new DoubleStatistics();
  /**
   * The constant cudaSetDeviceFlags_execution.
   */
  protected static final DoubleStatistics cudaSetDeviceFlags_execution = new DoubleStatistics();
  /**
   * The constant cudaMallocManaged_execution.
   */
  protected static final DoubleStatistics cudaMallocManaged_execution = new DoubleStatistics();
  /**
   * The constant cudaHostAlloc_execution.
   */
  protected static final DoubleStatistics cudaHostAlloc_execution = new DoubleStatistics();
  /**
   * The constant cudaFreeHost_execution.
   */
  protected static final DoubleStatistics cudaFreeHost_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceGetLimit_execution.
   */
  protected static final DoubleStatistics cudaDeviceGetLimit_execution = new DoubleStatistics();
  /**
   * The constant cudaDeviceSetLimit_execution.
   */
  protected static final DoubleStatistics cudaDeviceSetLimit_execution = new DoubleStatistics();
  
  /**
   * The constant cudaMemcpyAsync_execution.
   */
  protected static final DoubleStatistics cudaMemcpyAsync_execution = new DoubleStatistics();
  /**
   * The constant cudaMemcpy_execution.
   */
  protected static final DoubleStatistics cudaMemcpy_execution = new DoubleStatistics();
  /**
   * The constant cudaMemset_execution.
   */
  protected static final DoubleStatistics cudaMemset_execution = new DoubleStatistics();
  /**
   * The constant cudnnActivationBackward_execution.
   */
  protected static final DoubleStatistics cudnnActivationBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnActivationForward_execution.
   */
  protected static final DoubleStatistics cudnnActivationForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnAddTensor_execution.
   */
  protected static final DoubleStatistics cudnnAddTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardBias_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardBias_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardData_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardData_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionBackwardFilter_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionBackwardFilter_execution = new DoubleStatistics();
  /**
   * The constant cudnnConvolutionForward_execution.
   */
  protected static final DoubleStatistics cudnnConvolutionForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyActivationDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyActivationDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyConvolutionDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyConvolutionDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyFilterDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyFilterDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyOpTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyOpTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyPoolingDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyPoolingDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnDestroyTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnDestroyTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnGetPoolingNdForwardOutputDim_execution.
   */
  protected static final DoubleStatistics cudnnGetPoolingNdForwardOutputDim_execution = new DoubleStatistics();
  /**
   * The constant cudnnOpTensor_execution.
   */
  protected static final DoubleStatistics cudnnOpTensor_execution = new DoubleStatistics();
  /**
   * The constant cudnnPoolingBackward_execution.
   */
  protected static final DoubleStatistics cudnnPoolingBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnPoolingForward_execution.
   */
  protected static final DoubleStatistics cudnnPoolingForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnTransformTensor_execution.
   */
  protected static final DoubleStatistics cudnnTransformTensor_execution = new DoubleStatistics();
  /**
   * The constant deviceCount_execution.
   */
  protected static final DoubleStatistics deviceCount_execution = new DoubleStatistics();
  /**
   * The constant setDevice_execution.
   */
  protected static final DoubleStatistics setDevice_execution = new DoubleStatistics();
  /**
   * The constant getDeviceProperties_execution.
   */
  protected static final DoubleStatistics getDeviceProperties_execution = new DoubleStatistics();
  /**
   * The constant getOutputDims_execution.
   */
  protected static final DoubleStatistics getOutputDims_execution = new DoubleStatistics();
  /**
   * The constant newActivationDescriptor_execution.
   */
  protected static final DoubleStatistics newActivationDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newConvolutionNdDescriptor_execution.
   */
  protected static final DoubleStatistics newConvolutionNdDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newConvolutions2dDescriptor_execution.
   */
  protected static final DoubleStatistics newConvolutions2dDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newFilterDescriptor_execution.
   */
  protected static final DoubleStatistics newFilterDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newOpDescriptor_execution.
   */
  protected static final DoubleStatistics newOpDescriptor_execution = new DoubleStatistics();
  /**
   * The constant newTensorDescriptor_execution.
   */
  protected static final DoubleStatistics newTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant allocateBackwardDataWorkspace_execution.
   */
  protected static final DoubleStatistics allocateBackwardDataWorkspace_execution = new DoubleStatistics();
  /**
   * The constant allocateBackwardFilterWorkspace_execution.
   */
  protected static final DoubleStatistics allocateBackwardFilterWorkspace_execution = new DoubleStatistics();
  /**
   * The constant allocateForwardWorkspace_execution.
   */
  protected static final DoubleStatistics allocateForwardWorkspace_execution = new DoubleStatistics();
  /**
   * The constant getBackwardDataAlgorithm_execution.
   */
  protected static final DoubleStatistics getBackwardDataAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant getBackwardFilterAlgorithm_execution.
   */
  protected static final DoubleStatistics getBackwardFilterAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamCreate_execution.
   */
  protected static final DoubleStatistics cudaStreamCreate_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamDestroy_execution.
   */
  protected static final DoubleStatistics cudaStreamDestroy_execution = new DoubleStatistics();
  /**
   * The constant cudaStreamSynchronize_execution.
   */
  protected static final DoubleStatistics cudaStreamSynchronize_execution = new DoubleStatistics();
  /**
   * The constant getForwardAlgorithm_execution.
   */
  protected static final DoubleStatistics getForwardAlgorithm_execution = new DoubleStatistics();
  /**
   * The constant syncLock.
   */
  protected static final Object syncLock = new Object();
  private static final Executor garbageTruck = MoreExecutors.directExecutor();
  //Executors.newCachedThreadPool(new ThreadFactoryBuilder().setNameFormat("gpu-free-%d").setDaemon(true).build());
  /**
   * The constant gpuGeneration.
   */
  @javax.annotation.Nonnull
  public static AtomicInteger gpuGeneration = new AtomicInteger(0);
  
  
  /**
   * Instantiates a new Gpu system.
   */
  protected CudaSystem() {
  }
  
  /**
   * Log header.
   */
  public static void logHeader() {
    logger.info(getHeader());
  }
  
  /**
   * Gets header.
   *
   * @return the header
   */
  public static String getHeader() {
    return TestUtil.toString(CudaSystem::printHeader);
  }
  
  /**
   * Print header.
   *
   * @param out the out
   */
  public static void printHeader(@javax.annotation.Nonnull PrintStream out) {
    @javax.annotation.Nonnull int[] runtimeVersion = {0};
    @javax.annotation.Nonnull int[] driverVersion = {0};
    JCuda.cudaRuntimeGetVersion(runtimeVersion);
    JCuda.cudaDriverGetVersion(driverVersion);
    @javax.annotation.Nonnull String jCudaVersion = JCuda.getJCudaVersion();
    out.printf("Time: %s; Driver %s; Runtime %s; Lib %s%n", new Date(), driverVersion[0], runtimeVersion[0], jCudaVersion);
    @javax.annotation.Nonnull long[] free = {0};
    @javax.annotation.Nonnull long[] total = {0};
    JCuda.cudaMemGetInfo(free, total);
    out.printf("Cuda Memory: %.1f freeRef, %.1f total%n", free[0] * 1.0 / (1024 * 1024), total[0] * 1.0 / (1024 * 1024));
    @javax.annotation.Nonnull final int[] deviceCount = new int[1];
    jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    IntStream.range(0, deviceCount[0]).forEach(device -> {
      @javax.annotation.Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
      JCuda.cudaGetDeviceProperties(deviceProp, device);
      out.printf("Device %d = %s%n", device, deviceProp, free[0], total[0]);
    });
    System.getProperties().forEach((k, v) -> {
      boolean display = false;
      if (k.toString().endsWith(".version")) display = true;
      if (k.toString().startsWith("os.")) display = true;
      if (k.toString().contains("arch")) display = true;
      if (display) out.printf("%s = %s%n", k, v);
    });
  }
  
  /**
   * To map map.
   *
   * @param obj the obj
   * @return the map
   */
  @javax.annotation.Nonnull
  protected static Map<String, String> toMap(@javax.annotation.Nonnull DoubleStatistics obj) {
    @javax.annotation.Nonnull HashMap<String, String> map = new HashMap<>();
    if (0 < obj.getCount()) {
      map.put("stddev", Double.toString(obj.getStandardDeviation()));
      map.put("mean", Double.toString(obj.getAverage()));
      map.put("total", Double.toString(obj.getSum()));
      map.put("max", Double.toString(obj.getMax()));
      map.put("count", Double.toString(obj.getCount()));
    }
    return map;
  }
  
  /**
   * Gets execution statistics.
   *
   * @return the execution statistics
   */
  @javax.annotation.Nonnull
  public static final Map<String, Map<String, String>> getExecutionStatistics() {
    @javax.annotation.Nonnull HashMap<String, Map<String, String>> map = new HashMap<>();
    map.put("createPoolingDescriptor", toMap(createPoolingDescriptor_execution));
    map.put("cudaDeviceReset", toMap(cudaDeviceReset_execution));
    map.put("cudaFree", toMap(cudaFree_execution));
    map.put("cudaMalloc", toMap(cudaMalloc_execution));
    map.put("cudaMallocManaged", toMap(cudaMallocManaged_execution));
    map.put("cudaHostAlloc", toMap(cudaHostAlloc_execution));
    map.put("cudaFreeHost", toMap(cudaFreeHost_execution));
    map.put("cudaDeviceGetLimit", toMap(cudaDeviceGetLimit_execution));
    map.put("cudaDeviceSetLimit", toMap(cudaDeviceSetLimit_execution));
    map.put("cudaMemcpy", toMap(cudaMemcpy_execution));
    map.put("cudaMemset", toMap(cudaMemset_execution));
    map.put("cudnnActivationBackward", toMap(cudnnActivationBackward_execution));
    map.put("cudnnActivationForward", toMap(cudnnActivationForward_execution));
    map.put("cudnnAddTensor", toMap(cudnnAddTensor_execution));
    map.put("cudnnConvolutionBackwardBias", toMap(cudnnConvolutionBackwardBias_execution));
    map.put("cudnnConvolutionBackwardData", toMap(cudnnConvolutionBackwardData_execution));
    map.put("cudnnConvolutionBackwardFilter", toMap(cudnnConvolutionBackwardFilter_execution));
    map.put("cudnnConvolutionForward", toMap(cudnnConvolutionForward_execution));
    map.put("cudnnDestroyActivationDescriptor", toMap(cudnnDestroyActivationDescriptor_execution));
    map.put("cudnnDestroyConvolutionDescriptor", toMap(cudnnDestroyConvolutionDescriptor_execution));
    map.put("cudnnDestroyFilterDescriptor", toMap(cudnnDestroyFilterDescriptor_execution));
    map.put("cudnnDestroyOpTensorDescriptor", toMap(cudnnDestroyOpTensorDescriptor_execution));
    map.put("cudnnDestroyPoolingDescriptor", toMap(cudnnDestroyPoolingDescriptor_execution));
    map.put("cudnnDestroyTensorDescriptor", toMap(cudnnDestroyTensorDescriptor_execution));
    map.put("cudnnGetPoolingNdForwardOutputDim", toMap(cudnnGetPoolingNdForwardOutputDim_execution));
    map.put("cudnnOpTensor", toMap(cudnnOpTensor_execution));
    map.put("cudnnPoolingBackward", toMap(cudnnPoolingBackward_execution));
    map.put("cudnnPoolingForward", toMap(cudnnPoolingForward_execution));
    map.put("cudnnTransformTensor", toMap(cudnnTransformTensor_execution));
    map.put("deviceCount", toMap(deviceCount_execution));
    map.put("setDevice", toMap(setDevice_execution));
    map.put("getDeviceProperties", toMap(getDeviceProperties_execution));
    map.put("getOutputDims", toMap(getOutputDims_execution));
    map.put("newActivationDescriptor", toMap(newActivationDescriptor_execution));
    map.put("newConvolutionNdDescriptor", toMap(newConvolutionNdDescriptor_execution));
    map.put("newConvolutions2dDescriptor", toMap(newConvolutions2dDescriptor_execution));
    map.put("newFilterDescriptor", toMap(newFilterDescriptor_execution));
    map.put("newOpDescriptor", toMap(newOpDescriptor_execution));
    map.put("newTensorDescriptor", toMap(newTensorDescriptor_execution));
    map.put("allocateBackwardDataWorkspace", toMap(allocateBackwardDataWorkspace_execution));
    map.put("allocateBackwardFilterWorkspace", toMap(allocateBackwardFilterWorkspace_execution));
    map.put("allocateForwardWorkspace", toMap(allocateForwardWorkspace_execution));
    map.put("getBackwardDataAlgorithm", toMap(getBackwardDataAlgorithm_execution));
    map.put("getBackwardFilterAlgorithm", toMap(getBackwardFilterAlgorithm_execution));
    map.put("getForwardAlgorithm", toMap(getForwardAlgorithm_execution));
    map.put("cudaDeviceSynchronize", toMap(cudaDeviceSynchronize_execution));
    map.put("cudaStreamCreate", toMap(cudaStreamCreate_execution));
    map.put("cudaStreamDestroy", toMap(cudaStreamDestroy_execution));
    map.put("cudaStreamSynchronize", toMap(cudaStreamSynchronize_execution));
    map.put("cudaMemcpyAsync", toMap(cudaMemcpyAsync_execution));
    map.put("cudaSetDeviceFlags", toMap(cudaSetDeviceFlags_execution));
    
    for (String entry : map.entrySet().stream().filter(x -> x.getValue().isEmpty()).map(x -> x.getKey()).collect(Collectors.toList())) {
      map.remove(entry);
    }
    return map;
  }
  
  /**
   * Cuda device reset int.
   *
   * @return the int
   */
  public static int cudaDeviceReset() {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaDeviceReset();
    CudaSystem.log("cudaDeviceReset", result);
    cudaDeviceReset_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }
  
  /**
   * Cuda malloc int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @return the int
   */
  public static int cudaMalloc(final Pointer devPtr, final long size) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMalloc(devPtr, size);
    CudaSystem.log("cudaMalloc", result, devPtr, size);
    cudaMalloc_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }
  
  /**
   * Cuda malloc managed int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @param flags  the flags
   * @return the int
   */
  public static int cudaMallocManaged(final Pointer devPtr, final long size, int flags) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMallocManaged(devPtr, size, flags);
    CudaSystem.log("cudaMallocManaged", result, devPtr, size, flags);
    cudaMallocManaged_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }
  
  /**
   * Cuda device synchronize int.
   *
   * @return the int
   */
  public static int cudaDeviceSynchronize() {
    long startTime = System.nanoTime();
    synchronized (syncLock) {
      final int result = JCuda.cudaDeviceSynchronize();
      CudaSystem.log("cudaDeviceSynchronize", result);
      cudaDeviceSynchronize_execution.accept((System.nanoTime() - startTime) / 1e9);
      handle(result);
      return result;
    }
  }
  
  /**
   * Cuda set device flags int.
   *
   * @param flags the flags
   * @return the int
   */
  public static int cudaSetDeviceFlags(int flags) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaSetDeviceFlags(flags);
    CudaSystem.log("cudaSetDeviceFlags", result, flags);
    cudaDeviceSynchronize_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }
  
  /**
   * Cuda host alloc int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @param flags  the flags
   * @return the int
   */
  public static int cudaHostAlloc(final Pointer devPtr, final long size, int flags) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaHostAlloc(devPtr, size, flags);
    cudaHostAlloc_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaHostAlloc", result, devPtr, size, flags);
    handle(result);
    return result;
  }
  
  /**
   * Cuda freeRef host int.
   *
   * @param devPtr the dev ptr
   * @return the int
   */
  public static int cudaFreeHost(final Pointer devPtr) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaFreeHost(devPtr);
    cudaFreeHost_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaFreeHost", result, devPtr);
    handle(result);
    return result;
  }
  
  /**
   * Cuda device get limit long.
   *
   * @param limit the limit
   * @return the long
   */
  public static long cudaDeviceGetLimit(final int limit) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull long[] pValue = new long[1];
    final int result = JCuda.cudaDeviceGetLimit(pValue, limit);
    cudaDeviceGetLimit_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaDeviceGetLimit(", result, pValue, limit);
    return pValue[0];
  }
  
  /**
   * Cuda device set limit int.
   *
   * @param limit the limit
   * @param value the value
   * @return the int
   */
  public static void cudaDeviceSetLimit(final int limit, long value) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaDeviceSetLimit(limit, value);
    cudaDeviceSetLimit_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaDeviceSetLimit(", result, limit, value);
    handle(result);
  }
  
  /**
   * Cuda memcpy int.
   *
   * @param dst                 the dst
   * @param src                 the src
   * @param count               the count
   * @param cudaMemcpyKind_kind the cuda memcpy kind kind
   * @return the int
   */
  public static void cudaMemcpy(final Pointer dst, final Pointer src, final long count, final int cudaMemcpyKind_kind) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    cudaMemcpy_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaMemcpy", result, dst, src, count, cudaMemcpyKind_kind);
    handle(result);
  }
  
  /**
   * Cuda memcpy async.
   *
   * @param dst                 the dst
   * @param src                 the src
   * @param count               the count
   * @param cudaMemcpyKind_kind the cuda memcpy kind kind
   * @param stream              the stream
   */
  public static void cudaMemcpyAsync(final Pointer dst, final Pointer src, final long count, final int cudaMemcpyKind_kind, cudaStream_t stream) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemcpyAsync(dst, src, count, cudaMemcpyKind_kind, stream);
    cudaMemcpyAsync_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaMemcpyAsync", result, dst, src, count, cudaMemcpyKind_kind, stream);
    handle(result);
  }
  
  /**
   * Cuda stream create cuda resource.
   *
   * @return the cuda resource
   */
  public static CudaResource<cudaStream_t> cudaStreamCreate() {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull cudaStream_t stream = new cudaStream_t();
    int result = JCuda.cudaStreamCreate(stream);
    cudaStreamCreate_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaStreamCreate", result, stream);
    handle(result);
    return new CudaStream(stream);
  }
  
  /**
   * Cuda stream destroy int.
   *
   * @param stream the stream
   * @return the int
   */
  public static int cudaStreamDestroy(cudaStream_t stream) {
    long startTime = System.nanoTime();
    int result = JCuda.cudaStreamDestroy(stream);
    cudaStreamDestroy_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaStreamDestroy", result, stream);
    handle(result);
    return result;
  }
  
  /**
   * Cuda stream synchronize.
   *
   * @param stream the stream
   */
  public static void cudaStreamSynchronize(cudaStream_t stream) {
    long startTime = System.nanoTime();
    int result = JCuda.cudaStreamSynchronize(stream);
    cudaStreamSynchronize_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaStreamSynchronize", result, stream);
    handle(result);
  }
  
  /**
   * Cuda memset int.
   *
   * @param mem   the mem
   * @param c     the c
   * @param count the count
   * @return the int
   */
  public static void cudaMemset(final Pointer mem, final int c, final long count) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemset(mem, c, count);
    //cudaDeviceSynchronize();
    cudaMemset_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudaMemset", result, mem, c, count);
    handle(result);
  }
  
  /**
   * Cudnn destroy activation descriptor int.
   *
   * @param activationDesc the activation desc
   * @return the int
   */
  public static int cudnnDestroyActivationDescriptor(final cudnnActivationDescriptor activationDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyActivationDescriptor", result, activationDesc);
    return result;
  }
  
  /**
   * Cudnn destroy convolution descriptor int.
   *
   * @param convDesc the conv desc
   * @return the int
   */
  public static int cudnnDestroyConvolutionDescriptor(final cudnnConvolutionDescriptor convDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyConvolutionDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyConvolutionDescriptor", result, convDesc);
    return result;
  }
  
  /**
   * Cudnn destroy filter descriptor int.
   *
   * @param filterDesc the filter desc
   * @return the int
   */
  public static int cudnnDestroyFilterDescriptor(final cudnnFilterDescriptor filterDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyFilterDescriptor", result, filterDesc);
    return result;
  }
  
  /**
   * Cudnn destroy op tensor descriptor int.
   *
   * @param opTensorDesc the op tensor desc
   * @return the int
   */
  public static int cudnnDestroyOpTensorDescriptor(final cudnnOpTensorDescriptor opTensorDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyOpTensorDescriptor(opTensorDesc);
    cudnnDestroyOpTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyOpTensorDescriptor", result, opTensorDesc);
    return result;
  }
  
  /**
   * Cudnn destroy pooling descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @return the int
   */
  public static int cudnnDestroyPoolingDescriptor(final cudnnPoolingDescriptor poolingDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
    cudnnDestroyPoolingDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyPoolingDescriptor", result, poolingDesc);
    return result;
  }
  
  /**
   * Cudnn destroy tensor descriptor int.
   *
   * @param tensorDesc the tensor desc
   * @return the int
   */
  public static int cudnnDestroyTensorDescriptor(final cudnnTensorDescriptor tensorDesc) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnDestroyTensorDescriptor(tensorDesc);
    cudnnDestroyTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnDestroyTensorDescriptor", result, tensorDesc);
    return result;
  }
  
  /**
   * Cudnn get pooling nd forward output length int.
   *
   * @param poolingDesc      the pooling desc
   * @param inputTensorDesc  the input tensor desc
   * @param nbDims           the nb dims
   * @param outputTensorDimA the output tensor length a
   * @return the int
   */
  public static int cudnnGetPoolingNdForwardOutputDim(
    final cudnnPoolingDescriptor poolingDesc,
    final cudnnTensorDescriptor inputTensorDesc,
    final int nbDims,
    final int[] outputTensorDimA) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    cudnnGetPoolingNdForwardOutputDim_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnGetPoolingNdForwardOutputDim", result, poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    return result;
  }
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static int getThreadDevice() {
    final Integer integer = CudaSystem.currentDevice.get();
    return integer == null ? -1 : integer;
  }
  
  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final int[] deviceCount = new int[1];
    final int returnCode = jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    CudaSystem.log("cudaGetDeviceCount", returnCode, deviceCount);
    deviceCount_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.handle(returnCode);
    return deviceCount[0];
  }
  
  /**
   * Is oom boolean.
   *
   * @param t the t
   * @return the boolean
   */
  public static boolean isOom(final Throwable t) {
    if (t instanceof OutOfMemoryError) return true;
    //if (t instanceof com.simiacryptus.mindseye.lang.cudnn.CudaError) return true;
    if (null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  /**
   * Get stride int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  public static int[] getStride(@javax.annotation.Nonnull final int[] array) {
    return IntStream.range(0, array.length).map(i -> IntStream.range(i + 1, array.length).map(ii -> array[ii]).reduce((a, b) -> a * b).orElse(1)).toArray();
  }
  
  /**
   * Handle.
   *
   * @param returnCode the return code
   */
  public static void handle(final int returnCode) {
    if (returnCode != cudnnStatus.CUDNN_STATUS_SUCCESS) {
      CudaError cudaError = new CudaError("returnCode = " + cudnnStatus.stringFor(returnCode));
      logger.warn("Cuda Error", cudaError);
      throw cudaError;
    }
  }
  
  /**
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  protected static void log(final String method, final Object result, @Nullable final Object... args) {
    @javax.annotation.Nonnull final String paramString = null == args ? "" : Arrays.stream(args).map(CudaSystem::renderToLog).reduce((a, b) -> a + ", " + b).orElse("");
    final String message = String.format("%.6f @ %s(%d): %s(%s) = %s", (System.nanoTime() - CudaSystem.start) / 1e9, Thread.currentThread().getName(), currentDevice.get(), method, paramString, result);
    try {
      CudaSystem.apiLog.forEach(apiLog -> CudaSystem.logThread.submit(() -> apiLog.println(message)));
    } catch (ConcurrentModificationException e) {}
  }
  
  /**
   * Get output dims int [ ].
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @return the int [ ]
   */
  @javax.annotation.Nonnull
  public static int[] getOutputDims(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final int[] tensorOuputDims = new int[4];
    final int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    getOutputDims_execution.accept((System.nanoTime() - startTime) / 1e9);
    CudaSystem.log("cudnnGetConvolutionNdForwardOutputDim", result, convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    CudaSystem.handle(result);
    return tensorOuputDims;
  }
  
  /**
   * Render to log string.
   *
   * @param obj the obj
   * @return the string
   */
  protected static String renderToLog(final Object obj) {
    if (obj instanceof int[]) {
      if (((int[]) obj).length < 10) {
        return Arrays.toString((int[]) obj);
      }
    }
    if (obj instanceof double[]) {
      if (((double[]) obj).length < 10) {
        return Arrays.toString((double[]) obj);
      }
    }
    if (obj instanceof float[]) {
      if (((float[]) obj).length < 10) {
        return Arrays.toString((float[]) obj);
      }
    }
    if (obj instanceof long[]) {
      if (((long[]) obj).length < 10) {
        return Arrays.toString((long[]) obj);
      }
    }
    return obj.toString();
  }
  
  /**
   * Remove log boolean.
   *
   * @param apiLog the api log
   * @return the boolean
   */
  public static boolean removeLog(PrintStream apiLog) {
    return CudaSystem.apiLog.remove(apiLog);
  }
  
  /**
   * With device.
   *
   * @param n      the n
   * @param action the action
   */
  public static void withDevice(int n, @javax.annotation.Nonnull Runnable action) {
    final int currentDevice = getThreadDevice();
    try {
      CudaDevice.setDevice(n);
      action.run();
    } finally {
      if (currentDevice >= 0) CudaDevice.setDevice(currentDevice);
      else CudaSystem.currentDevice.remove();
    }
  }
  
  /**
   * With device t.
   *
   * @param <T>      the type parameter
   * @param deviceId the n
   * @param action   the action
   * @return the t
   */
  public static <T> T withDevice(int deviceId, @javax.annotation.Nonnull Supplier<T> action) {
    assert deviceId >= 0;
    final int currentDevice = getThreadDevice();
    try {
      CudaDevice.setDevice(deviceId);
      return action.get();
    } finally {
      if (currentDevice >= 0) CudaDevice.setDevice(currentDevice);
      else CudaSystem.currentDevice.remove();
    }
  }
  
  /**
   * Is enabled boolean.
   *
   * @return the boolean
   */
  public static boolean isEnabled() {
    return 0 < CudnnHandle.POOL.size();
  }
  
  /**
   * Add log.
   *
   * @param log the log
   */
  public static void addLog(@javax.annotation.Nonnull PrintStream log) {
    printHeader(log);
    apiLog.add(log);
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void run(@javax.annotation.Nonnull final Consumer<CudnnHandle> fn) {run(fn, true);}
  
  /**
   * Run.
   *
   * @param fn          the fn
   * @param synchronize the synchronize
   */
  public static void run(@javax.annotation.Nonnull final Consumer<CudnnHandle> fn, boolean synchronize) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    if (threadlocal != null) {
      try {
        threadlocal.initThread();
        fn.accept(threadlocal);
      } catch (@javax.annotation.Nonnull final RuntimeException e) {
        throw e;
      } catch (@javax.annotation.Nonnull final Exception e) {
        throw new RuntimeException(e);
      }
    }
    else {
      CudnnHandle.POOL.apply(gpu -> {
        try {
          CudnnHandle.threadContext.set(gpu);
          gpu.initThread();
          fn.accept(gpu);
        } catch (@javax.annotation.Nonnull final RuntimeException e) {
          throw e;
        } catch (@javax.annotation.Nonnull final Exception e) {
          throw new RuntimeException(e);
        } finally {
          cleanup(gpu, synchronize);
        }
      });
    }
  }
  
  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  public static <T> T eval(@javax.annotation.Nonnull final Function<CudnnHandle, T> fn) {return eval(fn, true);}
  
  /**
   * Call t.
   *
   * @param <T>         the type parameter
   * @param fn          the fn
   * @param synchronize the synchronize
   * @return the t
   */
  public static <T> T eval(@javax.annotation.Nonnull final Function<CudnnHandle, T> fn, boolean synchronize) {
    if (CudnnHandle.POOL.getAll().isEmpty()) {
      return fn.apply(new CudnnHandle(-1));
    }
    else {
      CudnnHandle threadlocal = CudnnHandle.threadContext.get();
      if (threadlocal != null) {
        try {
          threadlocal.initThread();
          T result = fn.apply(threadlocal);
          return result;
        } catch (@javax.annotation.Nonnull final RuntimeException e) {
          throw e;
        } catch (@javax.annotation.Nonnull final Exception e) {
          throw new RuntimeException(e);
        }
      }
      else {
        return CudnnHandle.POOL.run(gpu -> {
          try {
            CudnnHandle.threadContext.set(gpu);
            gpu.initThread();
            T result = fn.apply(gpu);
            return result;
          } catch (@javax.annotation.Nonnull final RuntimeException e) {
            throw e;
          } catch (@javax.annotation.Nonnull final Exception e) {
            throw new RuntimeException(e);
          } finally {
            cleanup(gpu, synchronize);
          }
        });
      }
    }
  }
  
  private static void cleanup(final CudnnHandle gpu, final boolean synchronize) {
    if (synchronize) CudaSystem.cudaDeviceSynchronize();
    CudnnHandle.threadContext.remove();
    ArrayList<CudaResourceBase> objsToFree = new ArrayList<>();
    gpu.cleanupNative.drainTo(objsToFree);
    objsToFree.stream().forEach(CudaResourceBase::release);
  }
}
