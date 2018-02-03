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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.data.DoubleStatistics;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaStream_t;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Main library wrapper class around the GpuSystem API, providing logging and managed wrappers.
 */
public class GpuSystem {
  
  /**
   * The constant INSTANCE.
   */
  public static final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The constant apiLog.
   */
  public static final HashSet<PrintStream> apiLog = new HashSet<>();
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(GpuSystem.class);
  /**
   * The constant propertyCache.
   */
  protected static final ConcurrentHashMap<Integer, cudaDeviceProp> propertyCache = new ConcurrentHashMap<>();
  /**
   * The constant memoryLimitInBytes.
   */
  protected static final int memoryLimitInBytes = 32 * 1024 * 1024;
  /**
   * The constant currentDevice.
   */
  protected static final ThreadLocal<Integer> currentDevice = new ThreadLocal<Integer>() {
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
  /**
   * The constant gpuGeneration.
   */
  public static AtomicInteger gpuGeneration = new AtomicInteger(0);
  
  /**
   * Instantiates a new Gpu system.
   */
  protected GpuSystem() {
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
    return TestUtil.toString(GpuSystem::printHeader);
  }
  
  /**
   * Print header.
   *
   * @param out the out
   */
  public static void printHeader(PrintStream out) {
    int[] runtimeVersion = {0};
    int[] driverVersion = {0};
    JCuda.cudaRuntimeGetVersion(runtimeVersion);
    JCuda.cudaDriverGetVersion(driverVersion);
    String jCudaVersion = JCuda.getJCudaVersion();
    out.printf("Time: %s; Driver %s; Runtime %s; Lib %s%n", new Date(), driverVersion[0], runtimeVersion[0], jCudaVersion);
    long[] free = {0};
    long[] total = {0};
    JCuda.cudaMemGetInfo(free, total);
    out.printf("Cuda Memory: %.1f freeRef, %.1f total%n", free[0] * 1.0 / (1024 * 1024), total[0] * 1.0 / (1024 * 1024));
    final int[] deviceCount = new int[1];
    jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    IntStream.range(0, deviceCount[0]).forEach(device -> {
      final cudaDeviceProp deviceProp = new cudaDeviceProp();
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
  protected static Map<String, String> toMap(DoubleStatistics obj) {
    HashMap<String, String> map = new HashMap<>();
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
   * Create pooling descriptor cuda resource.
   *
   * @param mode       the mode
   * @param poolDims   the pool dims
   * @param windowSize the window size
   * @param padding    the padding
   * @param stride     the stride
   * @return the cuda resource
   */
  public static CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(final int mode, final int poolDims, final int[] windowSize, final int[] padding, final int[] stride) {
    long startTime = System.nanoTime();
    final cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    GpuSystem.log("cudnnCreatePoolingDescriptor", result, poolingDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc,
                                                mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
                                                padding, stride);
    GpuSystem.log("cudnnSetPoolingNdDescriptor", result, poolingDesc,
                  mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
                  padding, stride);
    GpuSystem.handle(result);
    createPoolingDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, GpuSystem::cudnnDestroyPoolingDescriptor, getDevice());
  }
  
  /**
   * Gets execution statistics.
   *
   * @return the execution statistics
   */
  public static final Map<String, Map<String, String>> getExecutionStatistics() {
    HashMap<String, Map<String, String>> map = new HashMap<>();
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
    GpuSystem.log("cudaDeviceReset", result);
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
    GpuSystem.log("cudaMalloc", result, devPtr, size);
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
    GpuSystem.log("cudaMallocManaged", result, devPtr, size, flags);
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
      GpuSystem.log("cudaDeviceSynchronize", result);
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
    GpuSystem.log("cudaSetDeviceFlags", result, flags);
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
    GpuSystem.log("cudaHostAlloc", result, devPtr, size, flags);
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
    GpuSystem.log("cudaFreeHost", result, devPtr);
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
    long[] pValue = new long[1];
    final int result = JCuda.cudaDeviceGetLimit(pValue, limit);
    cudaDeviceGetLimit_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudaDeviceGetLimit(", result, pValue, limit);
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
    GpuSystem.log("cudaDeviceSetLimit(", result, limit, value);
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
    GpuSystem.log("cudaMemcpy", result, dst, src, count, cudaMemcpyKind_kind);
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
    GpuSystem.log("cudaMemcpyAsync", result, dst, src, count, cudaMemcpyKind_kind, stream);
    handle(result);
  }
  
  /**
   * Cuda stream create cuda resource.
   *
   * @return the cuda resource
   */
  public static CudaResource<cudaStream_t> cudaStreamCreate() {
    long startTime = System.nanoTime();
    cudaStream_t stream = new cudaStream_t();
    int result = JCuda.cudaStreamCreate(stream);
    cudaStreamCreate_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudaStreamCreate", result, stream);
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
    GpuSystem.log("cudaStreamDestroy", result, stream);
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
    GpuSystem.log("cudaStreamSynchronize", result, stream);
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
    cudaDeviceSynchronize();
    cudaMemset_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudaMemset", result, mem, c, count);
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
    GpuSystem.log("cudnnDestroyActivationDescriptor", result, activationDesc);
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
    GpuSystem.log("cudnnDestroyConvolutionDescriptor", result, convDesc);
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
    GpuSystem.log("cudnnDestroyFilterDescriptor", result, filterDesc);
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
    GpuSystem.log("cudnnDestroyOpTensorDescriptor", result, opTensorDesc);
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
    GpuSystem.log("cudnnDestroyPoolingDescriptor", result, poolingDesc);
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
    GpuSystem.log("cudnnDestroyTensorDescriptor", result, tensorDesc);
    return result;
  }
  
  /**
   * Cudnn get pooling nd forward output dim int.
   *
   * @param poolingDesc      the pooling desc
   * @param inputTensorDesc  the input tensor desc
   * @param nbDims           the nb dims
   * @param outputTensorDimA the output tensor dim a
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
    GpuSystem.log("cudnnGetPoolingNdForwardOutputDim", result, poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    return result;
  }
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static int getDevice() {
    final Integer integer = GpuSystem.currentDevice.get();
    return integer == null ? -1 : integer;
  }
  
  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    long startTime = System.nanoTime();
    final int[] deviceCount = new int[1];
    final int returnCode = jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    GpuSystem.log("cudaGetDeviceCount", returnCode, deviceCount);
    deviceCount_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.handle(returnCode);
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
    //if (t instanceof com.simiacryptus.mindseye.lang.cudnn.GpuError) return true;
    if (null != t.getCause() && t != t.getCause()) return isOom(t.getCause());
    return false;
  }
  
  /**
   * Get stride int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  public static int[] getStride(final int[] array) {
    return IntStream.range(0, array.length).map(i -> IntStream.range(i + 1, array.length).map(ii -> array[ii]).reduce((a, b) -> a * b).orElse(1)).toArray();
  }
  
  /**
   * Handle.
   *
   * @param returnCode the return code
   */
  public static void handle(final int returnCode) {
    if (returnCode != cudnnStatus.CUDNN_STATUS_SUCCESS) {
      throw new GpuError("returnCode = " + cudnnStatus.stringFor(returnCode));
    }
  }
  
  /**
   * Clean memory.
   *
   * @return the future
   */
  public static Future<?> cleanMemory() {
    return singleThreadExecutor.submit(() -> {
      try {
        logger.warn("Cleaning Memory");
        Runtime runtime = Runtime.getRuntime();
        RecycleBin.DOUBLES.clear();
        runtime.gc();
        GpuTensorList.evictAllToHeap();
        runtime.gc();
        runtime.runFinalization();
      } catch (Throwable e) {
        logger.warn("Error while cleaning memory", e);
      }
    });
  }
  
  /**
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  protected static void log(final String method, final Object result, final Object... args) {
    final String paramString = null == args ? "" : Arrays.stream(args).map(GpuSystem::renderToLog).reduce((a, b) -> a + ", " + b).orElse("");
    final String message = String.format("%.6f @ %s: %s(%s) = %s", (System.nanoTime() - GpuSystem.start) / 1e9, Thread.currentThread().getName(), method, paramString, result);
    try {
      GpuSystem.apiLog.forEach(apiLog -> GpuSystem.logThread.submit(() -> apiLog.println(message)));
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
  public static int[] getOutputDims(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc) {
    long startTime = System.nanoTime();
    final int[] tensorOuputDims = new int[4];
    final int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    getOutputDims_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionNdForwardOutputDim", result, convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    GpuSystem.handle(result);
    return tensorOuputDims;
  }
  
  /**
   * New activation descriptor cuda resource.
   *
   * @param mode     the mode
   * @param reluNan  the relu nan
   * @param reluCeil the relu ceil
   * @return the cuda resource
   */
  public static CudaResource<cudnnActivationDescriptor> newActivationDescriptor(final int mode, final int reluNan, final double reluCeil) {
    long startTime = System.nanoTime();
    final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    GpuSystem.log("cudnnCreateActivationDescriptor", result, desc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetActivationDescriptor", result, desc, mode, reluNan, reluCeil);
    GpuSystem.handle(result);
    return new CudaResource<>(desc, GpuSystem::cudnnDestroyActivationDescriptor, getDevice());
  }
  
  /**
   * New convolution nd descriptor cuda resource.
   *
   * @param mode     the mode
   * @param dataType the data type
   * @param padding  the padding
   * @param stride   the stride
   * @param dilation the dilation
   * @return the cuda resource
   */
  public static CudaResource<cudnnConvolutionDescriptor> newConvolutionNdDescriptor(final int mode, final int dataType, final int[] padding, final int[] stride, final int[] dilation) {
    long startTime = System.nanoTime();
    assert padding.length == stride.length;
    assert padding.length == dilation.length;
    assert Arrays.stream(padding).allMatch(x -> x >= 0);
    assert Arrays.stream(stride).allMatch(x -> x > 0);
    assert Arrays.stream(dilation).allMatch(x -> x > 0);
    final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    newConvolutionNdDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc,
                                                    3,
                                                    padding,
                                                    stride,
                                                    dilation,
                                                    mode,
                                                    dataType
                                                   );
    GpuSystem.log("cudnnSetConvolutionNdDescriptor", result, convDesc, padding.length,
                  padding,
                  stride,
                  dilation,
                  mode,
                  dataType);
    GpuSystem.handle(result);
    return new CudaResource<cudnnConvolutionDescriptor>(convDesc, GpuSystem::cudnnDestroyConvolutionDescriptor, getDevice()) {
      @Override
      public String toString() {
        return "cudnnSetConvolutionNdDescriptor(padding=" + Arrays.toString(padding) +
          ";stride=" + Arrays.toString(stride) +
          ";dilation=" + Arrays.toString(dilation) +
          ";mode=" + mode +
          ";dataType=" + dataType + ")";
      }
    };
  }
  
  /**
   * New convolutions 2 d descriptor cuda resource.
   *
   * @param mode         the mode
   * @param dataType     the data type
   * @param paddingY     the padding y
   * @param paddingX     the padding x
   * @param strideHeight the stride height
   * @param strideWidth  the stride width
   * @param dilationY    the dilation y
   * @param dilationX    the dilation x
   * @return the cuda resource
   */
  public static CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int mode, final int dataType, final int paddingY, final int paddingX, final int strideHeight, final int strideWidth, int dilationY, int dilationX) {
    long startTime = System.nanoTime();
    final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    GpuSystem.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetConvolution2dDescriptor(
      convDesc,
      paddingY, // zero-padding height
      paddingX, // zero-padding width
      strideHeight, // vertical filter stride
      strideWidth, // horizontal filter stride
      dilationY, // upscale the input in x-direction
      dilationX, // upscale the input in y-direction
      mode
      , dataType
                                                   );
    newConvolutions2dDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetConvolution2dDescriptor", result, convDesc,
                  paddingY, // zero-padding height
                  paddingX, // zero-padding width
                  strideHeight, // vertical filter stride
                  strideWidth, // horizontal filter stride
                  dilationY, // upscale the input in x-direction
                  dilationX, // upscale the input in y-direction
                  mode,
                  dataType);
    GpuSystem.handle(result);
    return new CudaResource<>(convDesc, GpuSystem::cudnnDestroyConvolutionDescriptor, getDevice());
  }
  
  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType       the data type
   * @param tensorLayout   the tensor layout
   * @param outputChannels the output channels
   * @param inputChannels  the input channels
   * @param height         the height
   * @param width          the width
   * @return the cuda resource
   */
  public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final int dataType, final int tensorLayout, final int outputChannels, final int inputChannels, final int height, final int width) {
    long startTime = System.nanoTime();
    final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    GpuSystem.log("cudnnCreateFilterDescriptor", result, filterDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetFilter4dDescriptor", result, filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    GpuSystem.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, GpuSystem::cudnnDestroyFilterDescriptor, getDevice()) {
      @Override
      public String toString() {
        return "cudnnSetFilter4dDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";outputChannels=" + outputChannels +
          ";inputChannels=" + inputChannels +
          ";height=" + height +
          ";=width" + width + ")";
      }
    };
  }
  
  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType     the data type
   * @param tensorLayout the tensor layout
   * @param dimensions   the dimensions
   * @return the cuda resource
   */
  public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(final int dataType, final int tensorLayout, final int[] dimensions) {
    long startTime = System.nanoTime();
    final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    GpuSystem.log("cudnnCreateFilterDescriptor", result, filterDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetFilterNdDescriptor", result, filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    GpuSystem.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, GpuSystem::cudnnDestroyFilterDescriptor, getDevice()) {
      @Override
      public String toString() {
        return "cudnnSetFilterNdDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";dimensions=" + Arrays.toString(dimensions) + ")";
      }
    };
  }
  
  /**
   * New op descriptor cuda resource.
   *
   * @param opType   the op type
   * @param dataType the data type
   * @return the cuda resource
   */
  public static CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(final int opType, final int dataType) {
    long startTime = System.nanoTime();
    final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    GpuSystem.log("cudnnCreateOpTensorDescriptor", result, opDesc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetOpTensorDescriptor", result, opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    GpuSystem.handle(result);
    return new CudaResource<>(opDesc, GpuSystem::cudnnDestroyOpTensorDescriptor, getDevice());
  }
  
  /**
   * New tensor descriptor cuda resource.
   *
   * @param dataType     the data type
   * @param tensorLayout the tensor layout
   * @param batchCount   the batch count
   * @param channels     the channels
   * @param height       the height
   * @param width        the width
   * @return the cuda resource
   */
  public static CudaResource<cudnnTensorDescriptor> newTensorDescriptor(final int dataType, final int tensorLayout,
                                                                        final int batchCount, final int channels, final int height, final int width) {
    long startTime = System.nanoTime();
    final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    GpuSystem.log("cudnnCreateTensorDescriptor", result, desc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetTensor4dDescriptor", result, desc, tensorLayout, dataType, batchCount, channels, height, width);
    GpuSystem.handle(result);
    return new CudaResource<cudnnTensorDescriptor>(desc, GpuSystem::cudnnDestroyTensorDescriptor, getDevice()) {
      @Override
      public String toString() {
        return "cudnnSetTensor4dDescriptor(dataType=" + dataType +
          ";tensorLayout=" + tensorLayout +
          ";batchCount=" + batchCount +
          ";channels=" + channels +
          ";height=" + height +
          ";=width" + width + ")";
      }
    };
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
    return GpuSystem.apiLog.remove(apiLog);
  }
  
  /**
   * New tensor descriptor cuda resource.
   *
   * @param dataType   the data type
   * @param batchCount the batch count
   * @param channels   the channels
   * @param height     the height
   * @param width      the width
   * @param nStride    the n stride
   * @param cStride    the c stride
   * @param hStride    the h stride
   * @param wStride    the w stride
   * @return the cuda resource
   */
  public static CudaResource<cudnnTensorDescriptor> newTensorDescriptor(final int dataType,
                                                                        final int batchCount, final int channels, final int height, final int width,
                                                                        final int nStride, final int cStride, final int hStride, final int wStride) {
    long startTime = System.nanoTime();
    final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    GpuSystem.log("cudnnCreateTensorDescriptor", result, desc);
    GpuSystem.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnSetTensor4dDescriptorEx", result, desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    GpuSystem.handle(result);
    return new CudaResource<>(desc, GpuSystem::cudnnDestroyTensorDescriptor, getDevice());
  }
  
  /**
   * Reset all GPUs and Heap Memory
   */
  public static void reset() {
    synchronized (GpuSystem.class) {
      cleanMemory();
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      try {
        IntStream.range(0, GpuSystem.deviceCount()).forEach(deviceNumber -> GpuSystem.withDevice(deviceNumber, () -> {
          logger.warn(String.format("Resetting Device %d", deviceNumber));
          CudaPtr.getGpuStats(deviceNumber).usedMemory.set(0);
          cudaDeviceReset();
        }));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      gpuGeneration.incrementAndGet();
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
  }
  
  /**
   * With device.
   *
   * @param n      the n
   * @param action the action
   */
  public static void withDevice(int n, Runnable action) {
    final int currentDevice = getDevice();
    try {
      GpuDevice.setDevice(n);
      action.run();
    } finally {
      if (currentDevice >= 0) GpuDevice.setDevice(currentDevice);
      else GpuSystem.currentDevice.remove();
    }
  }
  
  /**
   * With device t.
   *
   * @param <T>    the type parameter
   * @param n      the n
   * @param action the action
   * @return the t
   */
  public static <T> T withDevice(int n, Supplier<T> action) {
    if (n < 0) return action.get();
    final int currentDevice = getDevice();
    try {
      GpuDevice.setDevice(n);
      return action.get();
    } finally {
      if (currentDevice >= 0) GpuDevice.setDevice(currentDevice);
      else GpuSystem.currentDevice.remove();
    }
  }
  
  /**
   * Is enabled boolean.
   *
   * @return the boolean
   */
  public static boolean isEnabled() {
    return 0 < CuDNNHandle.POOL.size();
  }
  
  /**
   * Add log.
   *
   * @param log the log
   */
  public static void addLog(PrintStream log) {
    printHeader(log);
    apiLog.add(log);
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void run(final Consumer<CuDNNHandle> fn) {run(fn, true);}
  
  /**
   * Run.
   *
   * @param fn          the fn
   * @param synchronize the synchronize
   */
  public static void run(final Consumer<CuDNNHandle> fn, boolean synchronize) {
    CuDNNHandle threadlocal = CuDNNHandle.threadContext.get();
    try {
      if (threadlocal != null) {
        try {
          threadlocal.initThread();
          fn.accept(threadlocal);
        } catch (final RuntimeException e) {
          throw e;
        } catch (final Exception e) {
          throw new RuntimeException(e);
        }
      }
      else {
        CuDNNHandle.POOL.apply(exe -> {
          try {
            CuDNNHandle.threadContext.set(exe);
            exe.initThread();
            fn.accept(exe);
          } catch (final RuntimeException e) {
            throw e;
          } catch (final Exception e) {
            throw new RuntimeException(e);
          } finally {
            CuDNNHandle.threadContext.remove();
          }
        });
      }
    } finally {
      if (synchronize) GpuSystem.cudaDeviceSynchronize();
    }
  }
  
  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  public static <T> T eval(final Function<CuDNNHandle, T> fn) {return eval(fn, true);}
  
  /**
   * Call t.
   *
   * @param <T>         the type parameter
   * @param fn          the fn
   * @param synchronize the synchronize
   * @return the t
   */
  public static <T> T eval(final Function<CuDNNHandle, T> fn, boolean synchronize) {
    if (CuDNNHandle.POOL.getAll().isEmpty()) {
      return fn.apply(new CuDNNHandle(-1));
    }
    else {
      try {
        CuDNNHandle threadlocal = CuDNNHandle.threadContext.get();
        if (threadlocal != null) {
          try {
            threadlocal.initThread();
            T result = fn.apply(threadlocal);
            return result;
          } catch (final RuntimeException e) {
            throw e;
          } catch (final Exception e) {
            throw new RuntimeException(e);
          }
        }
        else {
          return CuDNNHandle.POOL.run(exe -> {
            try {
              CuDNNHandle.threadContext.set(exe);
              exe.initThread();
              T result = fn.apply(exe);
              return result;
            } catch (final RuntimeException e) {
              throw e;
            } catch (final Exception e) {
              throw new RuntimeException(e);
            } finally {
              CuDNNHandle.threadContext.remove();
            }
          });
        }
      } finally {
        if (synchronize) GpuSystem.cudaDeviceSynchronize();
        LinkedBlockingDeque<ReferenceCounting> deque = CuDNNHandle.CLEANUP.get();
        deque.stream().forEach(x -> x.freeRef());
        deque.clear();
      }
    }
  }
}
