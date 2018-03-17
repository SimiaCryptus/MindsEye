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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.lang.ResourcePool;
import com.simiacryptus.util.lang.StaticResourcePool;
import com.simiacryptus.util.lang.TimedResult;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnStatus;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaStream_t;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.ConcurrentModificationException;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Main library wrapper class around the CudaSystem API, providing logging and managed wrappers.
 */
public class CudaSystem {
  
  private static final Map<Integer, Long> syncTimes = new HashMap<>();
  
  /**
   * The constant INSTANCE.
   */
//  public static final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).getNetwork());
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
  protected static final ThreadLocal<Integer> currentDeviceId = new ThreadLocal<Integer>();
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
  protected static final DoubleStatistics cudnnSoftmaxForward_execution = new DoubleStatistics();
  /**
   * The constant cudnnActivationBackward_execution.
   */
  protected static final DoubleStatistics cudnnSoftmaxBackward_execution = new DoubleStatistics();
  /**
   * The constant cudnnCreateReduceTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnCreateReduceTensorDescriptor_execution = new DoubleStatistics();
  /**
   * The constant cudnnSetReduceTensorDescriptor_execution.
   */
  protected static final DoubleStatistics cudnnSetReduceTensorDescriptor_execution = new DoubleStatistics();
  
  
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
   * The constant cudnnReduceTensor_execution.
   */
  protected static final DoubleStatistics cudnnReduceTensor_execution = new DoubleStatistics();
  
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
  protected static final DoubleStatistics cudnnSetTensor_execution = new DoubleStatistics();
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
  //Executors.newCachedThreadPool(new ThreadFactoryBuilder().setNameFormat("gpu-free-%d").setDaemon(true).getNetwork());
  /**
   * The constant gpuGeneration.
   */
  @Nonnull
  public static AtomicInteger gpuGeneration = new AtomicInteger(0);
  private static volatile StaticResourcePool<CudnnHandle> pool;
//  private final List<StackTraceElement[]> dirty = new ArrayList<>();
  
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
  public static void printHeader(@Nonnull PrintStream out) {
    @Nonnull int[] runtimeVersion = {0};
    @Nonnull int[] driverVersion = {0};
    JCuda.cudaRuntimeGetVersion(runtimeVersion);
    JCuda.cudaDriverGetVersion(driverVersion);
    @Nonnull String jCudaVersion = JCuda.getJCudaVersion();
    out.printf("Time: %s; Driver %s; Runtime %s; Lib %s%n", new Date(), driverVersion[0], runtimeVersion[0], jCudaVersion);
    @Nonnull long[] free = {0};
    @Nonnull long[] total = {0};
    JCuda.cudaMemGetInfo(free, total);
    out.printf("Cuda Memory: %.1f freeRef, %.1f total%n", free[0] * 1.0 / (1024 * 1024), total[0] * 1.0 / (1024 * 1024));
    @Nonnull final int[] deviceCount = new int[1];
    JCuda.cudaGetDeviceCount(deviceCount);
    IntStream.range(0, deviceCount[0]).forEach(device -> {
      @Nonnull final cudaDeviceProp deviceProp = new cudaDeviceProp();
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
  @Nonnull
  protected static Map<String, String> toMap(@Nonnull DoubleStatistics obj) {
    @Nonnull HashMap<String, String> map = new HashMap<>();
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
  @Nonnull
  public static final Map<String, Map<String, String>> getExecutionStatistics() {
    @Nonnull HashMap<String, Map<String, String>> map = new HashMap<>();
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
    log("cudaDeviceReset", result, new Object[]{});
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
  public static int cudaMalloc(final CudaPointer devPtr, final long size) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMalloc(devPtr, size);
    log("cudaMalloc", result, new Object[]{devPtr, size});
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
  public static int cudaMallocManaged(final CudaPointer devPtr, final long size, int flags) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMallocManaged(devPtr, size, flags);
    log("cudaMallocManaged", result, new Object[]{devPtr, size, flags});
    cudaMallocManaged_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    return result;
  }
  
  /**
   * The constant handlePools.
   */
  protected static final HashMap<Integer, ResourcePool<CudnnHandle>> handlePools = new HashMap<>();
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static Integer getThreadDeviceId() {
    return CudaSystem.currentDeviceId.get();
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
    log("cudaSetDeviceFlags", result, new Object[]{flags});
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
  public static int cudaHostAlloc(final CudaPointer devPtr, final long size, int flags) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaHostAlloc(devPtr, size, flags);
    cudaHostAlloc_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaHostAlloc", result, new Object[]{devPtr, size, flags});
    handle(result);
    return result;
  }
  
  /**
   * Cuda freeRef host int.
   *
   * @param devPtr the dev ptr
   * @return the int
   */
  public static int cudaFreeHost(final CudaPointer devPtr) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaFreeHost(devPtr);
    cudaFreeHost_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaFreeHost", result, new Object[]{devPtr});
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
    @Nonnull long[] pValue = new long[1];
    final int result = JCuda.cudaDeviceGetLimit(pValue, limit);
    cudaDeviceGetLimit_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaDeviceGetLimit(", result, new Object[]{pValue, limit});
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
    log("cudaDeviceSetLimit(", result, new Object[]{limit, value});
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
  public static void cudaMemcpy(final CudaPointer dst, final CudaPointer src, final long count, final int cudaMemcpyKind_kind) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    cudaMemcpy_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaMemcpy", result, new Object[]{dst, src, count, cudaMemcpyKind_kind});
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
  public static void cudaMemcpyAsync(final CudaPointer dst, final CudaPointer src, final long count, final int cudaMemcpyKind_kind, cudaStream_t stream) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemcpyAsync(dst, src, count, cudaMemcpyKind_kind, stream);
    cudaMemcpyAsync_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaMemcpyAsync", result, new Object[]{dst, src, count, cudaMemcpyKind_kind, stream});
    handle(result);
  }
  
  /**
   * Cuda stream create cuda resource.
   *
   * @return the cuda resource
   */
  public static CudaResource<cudaStream_t> cudaStreamCreate() {
    long startTime = System.nanoTime();
    @Nonnull cudaStream_t stream = new cudaStream_t();
    int result = JCuda.cudaStreamCreate(stream);
    cudaStreamCreate_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaStreamCreate", result, new Object[]{stream});
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
    log("cudaStreamDestroy", result, new Object[]{stream});
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
    log("cudaStreamSynchronize", result, new Object[]{stream});
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
  public static void cudaMemset(final CudaPointer mem, final int c, final long count) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemset(mem, c, count);
    //cudaDeviceSynchronize();
    cudaMemset_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudaMemset", result, new Object[]{mem, c, count});
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
    log("cudnnDestroyActivationDescriptor", result, new Object[]{activationDesc});
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
    log("cudnnDestroyConvolutionDescriptor", result, new Object[]{convDesc});
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
    log("cudnnDestroyFilterDescriptor", result, new Object[]{filterDesc});
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
    log("cudnnDestroyOpTensorDescriptor", result, new Object[]{opTensorDesc});
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
    log("cudnnDestroyPoolingDescriptor", result, new Object[]{poolingDesc});
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
    log("cudnnDestroyTensorDescriptor", result, new Object[]{tensorDesc});
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
    log("cudnnGetPoolingNdForwardOutputDim", result, new Object[]{poolingDesc, inputTensorDesc, nbDims, outputTensorDimA});
    return result;
  }
  
  private static final int deviceCount = init();
  
  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    long startTime = System.nanoTime();
    @Nonnull final int[] deviceCount = new int[1];
    final int returnCode = JCuda.cudaGetDeviceCount(deviceCount);
    log("cudaGetDeviceCount", returnCode, new Object[]{deviceCount});
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
  public static int[] getStride(@Nonnull final int[] array) {
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
   * Get output dims int [ ].
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @return the int [ ]
   */
  @Nonnull
  public static int[] getOutputDims(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc) {
    long startTime = System.nanoTime();
    @Nonnull final int[] tensorOuputDims = new int[4];
    final int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    getOutputDims_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionNdForwardOutputDim", result, new Object[]{convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims});
    CudaSystem.handle(result);
    return tensorOuputDims;
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

//  /**
//   * With device.
//   *
//   * @param deviceId the n
//   * @param action   the action
//   */
//  public static void withDevice(int deviceId, @Nonnull Consumer<CudaDevice> action) {
//    assert deviceId >= 0;
//    final int prevDevice = getThreadDeviceId();
//    try {
//      CudaDevice.setDevice(deviceId);
//      action.accept(new CudaDevice(deviceId));
//    } finally {
//      if (prevDevice >= 0) CudaDevice.setDevice(prevDevice);
//      else CudaSystem.currentDeviceId.remove();
//    }
//  }
  
  
  /**
   * Add log.
   *
   * @param log the log
   */
  public static void addLog(@Nonnull PrintStream log) {
    printHeader(log);
    apiLog.add(log);
  }
  
  /**
   * The Execution thread.
   */
  protected final ExecutorService executionThread = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setNameFormat(toString()).build());
  
  /**
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  public static void log(final String method, final Object result, @Nullable final Object[] args) {
    String callstack = !CudaSettings.INSTANCE.isLogStack() ? "" : TestUtil.toString(Arrays.stream(Thread.currentThread().getStackTrace())
      .filter(x -> true
          && x.getClassName().startsWith("com.simiacryptus.mindseye.")
        //&& !x.getClassName().startsWith("com.simiacryptus.mindseye.lang.")
        //&& !x.getClassName().startsWith("com.simiacryptus.mindseye.test.")
      )
      //.limit(10)
      .toArray(i -> new StackTraceElement[i]), ", ");
    @Nonnull final String paramString = null == args ? "" : Arrays.stream(args).map(CudaSystem::renderToLog).reduce((a, b) -> a + ", " + b).orElse("");
    final String message = String.format("%.6f @ %s(%d): %s(%s) = %s via [%s]", (System.nanoTime() - CudaSystem.start) / 1e9, Thread.currentThread().getName(), getThreadDeviceId(), method, paramString, result, callstack);
    try {
      CudaSystem.apiLog.forEach(apiLog -> CudaSystem.logThread.submit(() -> apiLog.println(message)));
    } catch (ConcurrentModificationException e) {}
  }
  
  /**
   * Is thread device id boolean.
   *
   * @param deviceId the device id
   * @return the boolean
   */
  public static boolean isThreadDeviceId(int deviceId) {
    Integer integer = getThreadDeviceId();
    return integer != null && (deviceId == integer);
  }
  
  /**
   * Is enabled boolean.
   *
   * @return the boolean
   */
  public static boolean isEnabled() {
    return 0 < deviceCount;
  }
  
  /**
   * Run.
   *
   * @param deviceId the device id
   * @param fn       the fn
   */
  public static void withDevice(int deviceId, @Nonnull final Consumer<CudnnHandle> fn) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null && threadlocal.getDeviceId() == deviceId) {
        assert CudaSystem.isThreadDeviceId(threadlocal.getDeviceId());
        fn.accept(threadlocal);
      }
      else {
        getPool(deviceId).apply(gpu -> {
          gpu.wrap(() -> {
            fn.accept(gpu);
            return null;
          }).get();
        });
      }
    } finally {
      if (null == threadlocal) CudnnHandle.threadContext.remove();
      else CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice) CudaDevice.setDevice(incumbantDevice);
    }
  }
  
  /**
   * Run.
   *
   * @param <T>      the type parameter
   * @param deviceId the device id
   * @param action   the action
   * @return the t
   */
  public static <T> T withDevice(int deviceId, @Nonnull Function<CudnnHandle, T> action) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null && threadlocal.getDeviceId() == deviceId) {
        return action.apply(threadlocal);
      }
      else {
        return getPool(deviceId).apply(gpu -> {
          return gpu.wrap(() -> action.apply(gpu)).get();
        });
      }
    } finally {
      if (null == threadlocal) CudnnHandle.threadContext.remove();
      else CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice) CudaDevice.setDevice(incumbantDevice);
    }
  }
  
  /**
   * Run.
   *
   * @param fn    the fn
   * @param hints the hints
   */
  public static void run(@Nonnull final Consumer<CudnnHandle> fn, Object... hints) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null) {
        assert isThreadDeviceId(threadlocal.getDeviceId());
        fn.accept(threadlocal);
      }
      else {
        int device = chooseDevice(hints);
        getPool(device).apply(gpu -> {
          return gpu.wrap(() -> {
            fn.accept(gpu);
            return null;
          }).get();
        });
      }
    } finally {
      if (null == threadlocal) CudnnHandle.threadContext.remove();
      else CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice) CudaDevice.setDevice(incumbantDevice);
    }
  }
  
  /**
   * Gets thread handle.
   *
   * @return the thread handle
   */
  public static CudnnHandle getThreadHandle() {
    return CudnnHandle.threadContext.get();
  }
  
  /**
   * Call t.
   *
   * @param <T>   the type parameter
   * @param fn    the fn
   * @param hints the hints
   * @return the t
   */
  public static <T> T run(@Nonnull final Function<CudnnHandle, T> fn, Object... hints) {
    CudnnHandle threadlocal = CudnnHandle.threadContext.get();
    final Integer incumbantDevice = getThreadDeviceId();
    try {
      if (threadlocal != null) {
        assert CudaDevice.isThreadDeviceId(threadlocal.getDeviceId());
        T result = fn.apply(threadlocal);
        return result;
      }
      else {
        int device = chooseDevice(hints);
        assert device >= 0;
        return getPool(device).apply(gpu -> {
          return gpu.wrap(() -> fn.apply(gpu)).get();
        });
      }
    } finally {
      if (null == threadlocal) CudnnHandle.threadContext.remove();
      else CudnnHandle.threadContext.set(threadlocal);
      if (null != incumbantDevice) CudaDevice.setDevice(incumbantDevice);
    }
  }
  
  /**
   * Gets preference predicate.
   *
   * @param hints the hints
   * @return the preference predicate
   */
  public static int chooseDevice(final Object[] hints) {
    Set<Integer> devices = Arrays.stream(hints).map(hint -> {
      if (hint instanceof Result) {
        TensorList data = ((Result) hint).getData();
        if (data instanceof CudaTensorList) {
          int deviceId = ((CudaTensorList) data).getDeviceId();
          assert deviceId >= 0;
          return deviceId;
        }
      }
      else if (hint instanceof CudaDeviceResource) {
        int deviceId = ((CudaDeviceResource) hint).getDeviceId();
        //assert deviceId >= 0 : String.format("%s/%d", hint.getClass(), deviceId);
        if (deviceId >= 0) return deviceId;
      }
      else if (hint instanceof Integer) {
        Integer deviceId = (Integer) hint;
        assert deviceId >= 0;
        return deviceId;
      }
      return null;
    }).filter(x -> x != null).collect(Collectors.toSet());
    if (devices.isEmpty()) {
      int deviceId = (int) Math.floor(Math.random() * deviceCount);
      assert deviceId >= 0;
      return deviceId;
    }
    else {
      Integer deviceId = devices.stream().findAny().get();
      assert deviceId >= 0;
      return deviceId;
    }
  }
  
  /**
   * Load gpu contexts list. If the property disableCuDnn is set to true, no GPUs will be recognized. This is useful for
   * testing CPU-only compatibility.
   *
   * @return the list
   */
  static int init() {
    if (CudaSettings.INSTANCE.isDisable()) {
      CudaDevice.logger.warn("Disabled CudaSystem");
    }
    final int deviceCount;
    deviceCount = getDeviceCount();
    for (int d = 0; d < deviceCount; d++) {
      initDevice(d);
    }
    return deviceCount;
  }
  
  private static void initDevice(final int deviceNumber) {
    CudaDevice.setDevice(deviceNumber);
    CudaDevice.logger.info(String.format("Device %s - %s", deviceNumber, CudaDevice.getDeviceName(deviceNumber)));
    try {
      //CudaSystem.handle(CudaSystem.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync));
    } catch (Throwable e) {
      CudaDevice.logger.warn("Error initializing GPU", e);
      throw new RuntimeException(e);
    }
    for (@Nonnull DeviceLimits limit : DeviceLimits.values()) {
      CudaDevice.logger.info(String.format("Default Limit %s = %s", limit, limit.get()));
    }
    DeviceLimits.HeapSize.set(16 * 1024 * 1024 * 1024);
    DeviceLimits.FifoSize.set(8 * 1024 * 1024);
    for (@Nonnull DeviceLimits limit : DeviceLimits.values()) {
      CudaDevice.logger.info(String.format("Configured Limit %s = %s", limit, limit.get()));
    }
  }
  
  private static int getDeviceCount() {
    final int deviceCount;
    if (CudaSettings.INSTANCE.isForceSingleGpu()) {
      CudaDevice.logger.warn("Forcing Single-GPU Mode");
      deviceCount = 1;
    }
    else {
      deviceCount = CudaSystem.deviceCount();
    }
    CudaDevice.logger.info(String.format("Found %s devices", deviceCount));
    return deviceCount;
  }
  
  private static final HashMap<Integer, Object> deviceLocks = new HashMap<>();
  
  /**
   * Synchronize.
   *
   * @param time   the time
   * @param device the device
   */
  public static void synchronize(long time, int device) {
    long startTime = System.nanoTime();
    Long val = syncTimes.get(device);
    if (null == val) val = 0L;
    if (null == val || val < time) {
      final Long finalVal = val;
      String caller = !CudaSettings.INSTANCE.isProfileMemoryIO() ? "" : TestUtil.getCaller();
      withDevice(device, gpu -> {
        if (null == finalVal || finalVal < time) {
          synchronized (deviceLocks.computeIfAbsent(device, d -> new Object())) {
            if (null == finalVal || finalVal < time) {
              TimedResult<Long> timedResult = TimedResult.time(() -> cudaDeviceSynchronize());
              CudaTensorList.logger.debug(String.format("Synchronized %d in %.4f (%.6f -> %.6f -> %.6f) via %s", getThreadDeviceId(), timedResult.seconds(), (finalVal - startTime) / 1e9, (time - startTime) / 1e9, (timedResult.result - startTime) / 1e9, caller));
            }
          }
        }
      });
    }
  }
  
  /**
   * Cuda device synchronize int.
   *
   * @return the int
   */
  public static long cudaDeviceSynchronize() {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaDeviceSynchronize();
    log("cudaDeviceSynchronize", result, new Object[]{});
    cudaDeviceSynchronize_execution.accept((System.nanoTime() - startTime) / 1e9);
    handle(result);
    syncTimes.put(getThreadDeviceId(), startTime);
    return startTime;
  }
  
  /**
   * Cleanup.
   */
  protected void cleanup() {
    CudnnHandle.threadContext.remove();
  }
  
  /**
   * The interface Cuda device resource.
   */
  public interface CudaDeviceResource {
    /**
     * Gets device id.
     *
     * @return the device id
     */
    int getDeviceId();
  }
  
  /**
   * The constant POOL.
   *
   * @param deviceId the device id
   * @return the pool
   */
  public static ResourcePool<CudnnHandle> getPool(final int deviceId) {
    assert deviceId >= 0;
    return handlePools.computeIfAbsent(deviceId, d -> new ResourcePool<CudnnHandle>(32) {
      @Override
      public CudnnHandle create() {
        return new CudnnHandle(deviceId);
      }
    });
  }
}
