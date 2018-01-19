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
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.lang.StaticResourcePool;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Main library wrapper class around the CuDNN API, providing logging and managed wrappers.
 */
public class CuDNN {
  /**
   * The constant INSTANCE.
   */
  public static final ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  /**
   * The constant apiLog.
   */
  public static final HashSet<PrintStream> apiLog = new HashSet<>();
  private static final ConcurrentHashMap<Integer, cudaDeviceProp> propertyCache = new ConcurrentHashMap<>();
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CuDNN.class);
  private static final ThreadLocal<Integer> currentDevice = new ThreadLocal<Integer>() {
    @Override
    protected Integer initialValue() {
      return -1;
    }
  };
  private static final ExecutorService logThread = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  private static final long start = System.nanoTime();
  private static final DoubleStatistics createPoolingDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaDeviceReset_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaFree_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaMalloc_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaMallocManaged_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaHostAlloc_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaFreeHost_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaDeviceGetLimit_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaDeviceSetLimit_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaMemcpy_execution = new DoubleStatistics();
  private static final DoubleStatistics cudaMemset_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnActivationBackward_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnActivationForward_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnAddTensor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnConvolutionBackwardBias_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnConvolutionBackwardData_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnConvolutionBackwardFilter_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnConvolutionForward_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyActivationDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyConvolutionDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyFilterDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyOpTensorDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyPoolingDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnDestroyTensorDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnGetPoolingNdForwardOutputDim_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnOpTensor_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnPoolingBackward_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnPoolingForward_execution = new DoubleStatistics();
  private static final DoubleStatistics cudnnTransformTensor_execution = new DoubleStatistics();
  private static final DoubleStatistics deviceCount_execution = new DoubleStatistics();
  private static final DoubleStatistics setDevice_execution = new DoubleStatistics();
  private static final DoubleStatistics getDeviceProperties_execution = new DoubleStatistics();
  private static final DoubleStatistics getOutputDims_execution = new DoubleStatistics();
  private static final DoubleStatistics newActivationDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics newConvolutionNdDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics newConvolutions2dDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics newFilterDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics newOpDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics newTensorDescriptor_execution = new DoubleStatistics();
  private static final DoubleStatistics allocateBackwardDataWorkspace_execution = new DoubleStatistics();
  private static final DoubleStatistics allocateBackwardFilterWorkspace_execution = new DoubleStatistics();
  private static final DoubleStatistics allocateForwardWorkspace_execution = new DoubleStatistics();
  private static final DoubleStatistics getBackwardDataAlgorithm_execution = new DoubleStatistics();
  private static final DoubleStatistics getBackwardFilterAlgorithm_execution = new DoubleStatistics();
  private static final DoubleStatistics getForwardAlgorithm_execution = new DoubleStatistics();
  private static final ThreadLocal<CuDNN> threadContext = new ThreadLocal<>();
  //  /**
//   * Cudnn create reduce tensor descriptor int.
//   *
//   * @param reduceTensorDesc the reduce tensor desc
//   * @return the int
//   */
//  public static int cudnnCreateReduceTensorDescriptor(
//    final cudnnReduceTensorDescriptor reduceTensorDesc) {
//    final int result = JCudnn.cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
//    CuDNN.log("cudnnCreateReduceTensorDescriptor", result, reduceTensorDesc);
//    CuDNN.handle(result);
//    return result;
//  }
  private final int memoryLimitInBytes = 32 * 1024 * 1024;
  
  /**
   * The constant gpuContexts.
   */
  public static StaticResourcePool<CuDNN> contexts = new StaticResourcePool<>(loadGpuContexts());
  /**
   * The Cudnn handle.
   */
  public final cudnnHandle cudnnHandle;
  private final String deviceName;
  private final int deviceNumber;
  
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
    for (String entry : map.entrySet().stream().filter(x -> x.getValue().isEmpty()).map(x -> x.getKey()).collect(Collectors.toList())) {
      map.remove(entry);
    }
    return map;
  }
  
  /**
   * Instantiates a new Cu dnn.
   *
   * @param deviceNumber the device number
   */
  protected CuDNN(final int deviceNumber) {
    this.deviceNumber = deviceNumber;
    if (0 <= this.deviceNumber) {
      cudnnHandle = new cudnnHandle();
      initThread();
      deviceName = CuDNN.getDeviceName(deviceNumber);
      JCudnn.cudnnCreate(cudnnHandle);
    }
    else {
      cudnnHandle = null;
      deviceName = null;
    }
    //cudaSetDevice();
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
    return TestUtil.toString(CuDNN::printHeader);
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
    out.printf("Cuda Memory: %.1f free, %.1f total%n", free[0] * 1.0 / (1024 * 1024), total[0] * 1.0 / (1024 * 1024));
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
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @return the cuda ptr
   */
  public static CudaPtr alloc(final int deviceId, final long size) {return alloc(deviceId, size, false);}
  
  private static Map<String, String> toMap(DoubleStatistics obj) {
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
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @param dirty    the dirty
   * @return the cuda ptr
   */
  public static CudaPtr alloc(final int deviceId, final long size, boolean dirty) {return alloc(deviceId, size, MemoryType.Device, dirty);}
  
  /**
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @param type     the type
   * @return the cuda ptr
   */
  public static CudaPtr alloc(final int deviceId, final long size, MemoryType type) {return alloc(deviceId, size, type, false);}
  
  /**
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @param type     the type
   * @param dirty    the dirty
   * @return the cuda ptr
   */
  public static CudaPtr alloc(final int deviceId, final long size, MemoryType type, boolean dirty) {
    return new CudaPtr(size, deviceId, type, dirty);
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
    CuDNN.log("cudnnCreatePoolingDescriptor", result, poolingDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc,
                                                mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
                                                padding, stride);
    CuDNN.log("cudnnSetPoolingNdDescriptor", result, poolingDesc,
              mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
              padding, stride);
    CuDNN.handle(result);
    createPoolingDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    return new CudaResource<>(poolingDesc, CuDNN::cudnnDestroyPoolingDescriptor, getDevice());
  }
  
  /**
   * Cuda device reset int.
   *
   * @return the int
   */
  public static int cudaDeviceReset() {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaDeviceReset();
    CuDNN.log("cudaDeviceReset", result);
    cudaDeviceReset_execution.accept((System.nanoTime() - startTime) / 1e9);
    return result;
  }
  
  /**
   * Cuda free int.
   *
   * @param devPtr   the dev ptr
   * @param deviceId the device id
   * @return the int
   */
  public static int cudaFree(final Pointer devPtr, int deviceId) {
    long startTime = System.nanoTime();
    return CuDNN.withDevice(deviceId, () -> {
      final int result = JCuda.cudaFree(devPtr);
      CuDNN.log("cudaFree", result, devPtr);
      cudaFree_execution.accept((System.nanoTime() - startTime) / 1e9);
      return result;
    });
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
    CuDNN.log("cudaMalloc", result, devPtr, size);
    cudaMalloc_execution.accept((System.nanoTime() - startTime) / 1e9);
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
    CuDNN.log("cudaMallocManaged", result, devPtr, size, flags);
    cudaMallocManaged_execution.accept((System.nanoTime() - startTime) / 1e9);
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
    CuDNN.log("cudaHostAlloc", result, devPtr, size, flags);
    return result;
  }
  
  /**
   * Cuda free host int.
   *
   * @param devPtr the dev ptr
   * @return the int
   */
  public static int cudaFreeHost(final Pointer devPtr) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaFreeHost(devPtr);
    cudaFreeHost_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudaFreeHost", result, devPtr);
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
    CuDNN.log("cudaDeviceGetLimit(", result, pValue, limit);
    handle(result);
    return pValue[0];
  }
  
  /**
   * Cuda device set limit int.
   *
   * @param limit the limit
   * @param value the value
   * @return the int
   */
  public static int cudaDeviceSetLimit(final int limit, long value) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaDeviceSetLimit(limit, value);
    cudaDeviceSetLimit_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudaDeviceSetLimit(", result, limit, value);
    return result;
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
  public static int cudaMemcpy(final Pointer dst, final Pointer src, final long count, final int cudaMemcpyKind_kind) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    cudaMemcpy_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudaMemcpy", result, dst, src, count, cudaMemcpyKind_kind);
    return result;
  }

//  /**
//   * Cudnn reduce tensor int.
//   *
//   * @param handle               the handle
//   * @param reduceTensorDesc     the reduce tensor desc
//   * @param indices              the indices
//   * @param indicesSizeInBytes   the indices size in bytes
//   * @param workspace            the workspace
//   * @param workspaceSizeInBytes the workspace size in bytes
//   * @param alpha                the alpha
//   * @param aDesc                the a desc
//   * @param A                    the a
//   * @param beta                 the beta
//   * @param cDesc                the c desc
//   * @param C                    the c
//   * @return the int
//   */
//  public static int cudnnReduceTensor(
//    final cudnnHandle handle,
//    final cudnnReduceTensorDescriptor reduceTensorDesc,
//    final Pointer indices,
//    final long indicesSizeInBytes,
//    final Pointer workspace,
//    final long workspaceSizeInBytes,
//    final Pointer alpha,
//    final cudnnTensorDescriptor aDesc,
//    final Pointer A,
//    final Pointer beta,
//    final cudnnTensorDescriptor cDesc,
//    final Pointer C) {
//    final int result = JCudnn.cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
//    CuDNN.log("cudnnReduceTensor", result, handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
//    CuDNN.handle(result);
//    return result;
//  }
//
//  /**
//   * Cudnn setByCoord reduce tensor descriptor int.
//   *
//   * @param reduceTensorDesc        the reduce tensor desc
//   * @param reduceTensorOp          the reduce tensor op
//   * @param reduceTensorCompType    the reduce tensor comp type
//   * @param reduceTensorNanOpt      the reduce tensor nan opt
//   * @param reduceTensorIndices     the reduce tensor indices
//   * @param reduceTensorIndicesType the reduce tensor indices type
//   * @return the int
//   */
//  public static int cudnnSetReduceTensorDescriptor(
//    final cudnnReduceTensorDescriptor reduceTensorDesc,
//    final int reduceTensorOp,
//    final int reduceTensorCompType,
//    final int reduceTensorNanOpt,
//    final int reduceTensorIndices,
//    final int reduceTensorIndicesType) {
//    final int result = JCudnn.cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
//    CuDNN.log("cudnnSetReduceTensorDescriptor", result, reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
//    CuDNN.handle(result);
//    return result;
//  }
  
  /**
   * Cuda memset int.
   *
   * @param mem   the mem
   * @param c     the c
   * @param count the count
   * @return the int
   */
  public static int cudaMemset(final Pointer mem, final int c, final long count) {
    long startTime = System.nanoTime();
    final int result = JCuda.cudaMemset(mem, c, count);
    cudaMemset_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudaMemset", result, mem, c, count);
    return result;
  }
  
  /**
   * Cudnn activation backward int.
   *
   * @param handle         the handle
   * @param activationDesc the activation desc
   * @param alpha          the alpha
   * @param yDesc          the y desc
   * @param y              the y
   * @param dyDesc         the dy desc
   * @param dy             the dy
   * @param xDesc          the x desc
   * @param x              the x
   * @param beta           the beta
   * @param dxDesc         the dx desc
   * @param dx             the dx
   * @return the int
   */
  public static int cudnnActivationBackward(
    final cudnnHandle handle,
    final cudnnActivationDescriptor activationDesc,
    final Pointer alpha,
    final cudnnTensorDescriptor yDesc,
    final Pointer y,
    final cudnnTensorDescriptor dyDesc,
    final Pointer dy,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final Pointer beta,
    final cudnnTensorDescriptor dxDesc,
    final Pointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    cudnnActivationBackward_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnActivationBackward", result, handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    return result;
  }
  
  /**
   * Cudnn activation forward int.
   *
   * @param handle         the handle
   * @param activationDesc the activation desc
   * @param alpha          the alpha
   * @param xDesc          the x desc
   * @param x              the x
   * @param beta           the beta
   * @param yDesc          the y desc
   * @param y              the y
   * @return the int
   */
  public static int cudnnActivationForward(
    final cudnnHandle handle,
    final cudnnActivationDescriptor activationDesc,
    final Pointer alpha,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final Pointer beta,
    final cudnnTensorDescriptor yDesc,
    final Pointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnActivationForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnActivationForward", result, handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    return result;
  }
  
  /**
   * Cudnn add tensor int.
   *
   * @param handle the handle
   * @param alpha  the alpha
   * @param aDesc  the a desc
   * @param A      the a
   * @param beta   the beta
   * @param cDesc  the c desc
   * @param C      the c
   * @return the int
   */
  public static int cudnnAddTensor(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnTensorDescriptor aDesc,
    final Pointer A,
    final Pointer beta,
    final cudnnTensorDescriptor cDesc,
    final Pointer C) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
    cudnnAddTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnAddTensor", result, handle, alpha, aDesc, A, beta, cDesc, C);
    CuDNN.handle(result);
    return result;
  }
  
  /**
   * Cudnn convolution backward bias int.
   *
   * @param handle the handle
   * @param alpha  the alpha
   * @param dyDesc the dy desc
   * @param dy     the dy
   * @param beta   the beta
   * @param dbDesc the db desc
   * @param db     the db
   * @return the int
   */
  public static int cudnnConvolutionBackwardBias(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnTensorDescriptor dyDesc,
    final Pointer dy,
    final Pointer beta,
    final cudnnTensorDescriptor dbDesc,
    final Pointer db) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
    cudnnConvolutionBackwardBias_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnConvolutionBackwardBias", result, handle, alpha, dyDesc, dy, beta, dbDesc, db);
    return result;
  }
  
  /**
   * Cudnn convolution backward data int.
   *
   * @param handle               the handle
   * @param alpha                the alpha
   * @param wDesc                the w desc
   * @param w                    the w
   * @param dyDesc               the dy desc
   * @param dy                   the dy
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param dxDesc               the dx desc
   * @param dx                   the dx
   * @return the int
   */
  public static int cudnnConvolutionBackwardData(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnFilterDescriptor wDesc,
    final Pointer w,
    final cudnnTensorDescriptor dyDesc,
    final Pointer dy,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final Pointer workSpace,
    final long workSpaceSizeInBytes,
    final Pointer beta,
    final cudnnTensorDescriptor dxDesc,
    final Pointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    cudnnConvolutionBackwardData_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnConvolutionBackwardData", result, handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    return result;
  }
  
  /**
   * Cudnn convolution backward filter int.
   *
   * @param handle               the handle
   * @param alpha                the alpha
   * @param xDesc                the x desc
   * @param x                    the x
   * @param dyDesc               the dy desc
   * @param dy                   the dy
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param dwDesc               the dw desc
   * @param dw                   the dw
   * @return the int
   */
  public static int cudnnConvolutionBackwardFilter(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final cudnnTensorDescriptor dyDesc,
    final Pointer dy,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final Pointer workSpace,
    final long workSpaceSizeInBytes,
    final Pointer beta,
    final cudnnFilterDescriptor dwDesc,
    final Pointer dw) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    cudnnConvolutionBackwardFilter_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnConvolutionBackwardFilter", result, handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    return result;
  }
  
  /**
   * Cudnn convolution forward int.
   *
   * @param handle               the handle
   * @param alpha                the alpha
   * @param xDesc                the x desc
   * @param x                    the x
   * @param wDesc                the w desc
   * @param w                    the w
   * @param convDesc             the conv desc
   * @param algo                 the algo
   * @param workSpace            the work space
   * @param workSpaceSizeInBytes the work space size in bytes
   * @param beta                 the beta
   * @param yDesc                the y desc
   * @param y                    the y
   * @return the int
   */
  public static int cudnnConvolutionForward(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final cudnnFilterDescriptor wDesc,
    final Pointer w,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final Pointer workSpace,
    final long workSpaceSizeInBytes,
    final Pointer beta,
    final cudnnTensorDescriptor yDesc,
    final Pointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    cudnnConvolutionForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnConvolutionForward", result, handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    return result;
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
    CuDNN.log("cudnnDestroyActivationDescriptor", result, activationDesc);
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
    CuDNN.log("cudnnDestroyConvolutionDescriptor", result, convDesc);
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
    CuDNN.log("cudnnDestroyFilterDescriptor", result, filterDesc);
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
    CuDNN.log("cudnnDestroyOpTensorDescriptor", result, opTensorDesc);
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
    CuDNN.log("cudnnDestroyPoolingDescriptor", result, poolingDesc);
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
    CuDNN.log("cudnnDestroyTensorDescriptor", result, tensorDesc);
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
    CuDNN.log("cudnnGetPoolingNdForwardOutputDim", result, poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    return result;
  }
  
  /**
   * Cudnn op tensor int.
   *
   * @param handle       the handle
   * @param opTensorDesc the op tensor desc
   * @param alpha1       the alpha 1
   * @param aDesc        the a desc
   * @param A            the a
   * @param alpha2       the alpha 2
   * @param bDesc        the b desc
   * @param B            the b
   * @param beta         the beta
   * @param cDesc        the c desc
   * @param C            the c
   * @return the int
   */
  public static int cudnnOpTensor(
    final cudnnHandle handle,
    final cudnnOpTensorDescriptor opTensorDesc,
    final Pointer alpha1,
    final cudnnTensorDescriptor aDesc,
    final Pointer A,
    final Pointer alpha2,
    final cudnnTensorDescriptor bDesc,
    final Pointer B,
    final Pointer beta,
    final cudnnTensorDescriptor cDesc,
    final Pointer C) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    cudnnOpTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnOpTensor", result, handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    return result;
  }
  
  /**
   * Cudnn pooling backward int.
   *
   * @param handle      the handle
   * @param poolingDesc the pooling desc
   * @param alpha       the alpha
   * @param yDesc       the y desc
   * @param y           the y
   * @param dyDesc      the dy desc
   * @param dy          the dy
   * @param xDesc       the x desc
   * @param x           the x
   * @param beta        the beta
   * @param dxDesc      the dx desc
   * @param dx          the dx
   * @return the int
   */
  public static int cudnnPoolingBackward(
    final cudnnHandle handle,
    final cudnnPoolingDescriptor poolingDesc,
    final Pointer alpha,
    final cudnnTensorDescriptor yDesc,
    final Pointer y,
    final cudnnTensorDescriptor dyDesc,
    final Pointer dy,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final Pointer beta,
    final cudnnTensorDescriptor dxDesc,
    final Pointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    cudnnPoolingBackward_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnPoolingBackward", result, handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    return result;
  }
  
  /**
   * Cudnn pooling forward int.
   *
   * @param handle      the handle
   * @param poolingDesc the pooling desc
   * @param alpha       the alpha
   * @param xDesc       the x desc
   * @param x           the x
   * @param beta        the beta
   * @param yDesc       the y desc
   * @param y           the y
   * @return the int
   */
  public static int cudnnPoolingForward(
    final cudnnHandle handle,
    final cudnnPoolingDescriptor poolingDesc,
    final Pointer alpha,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final Pointer beta,
    final cudnnTensorDescriptor yDesc,
    final Pointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnPoolingForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnPoolingForward", result, handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    return result;
  }
  
  /**
   * Cudnn transform tensor int.
   *
   * @param handle the handle
   * @param alpha  the alpha
   * @param xDesc  the x desc
   * @param x      the x
   * @param beta   the beta
   * @param yDesc  the y desc
   * @param y      the y
   * @return the int
   */
  public static int cudnnTransformTensor(
    final cudnnHandle handle,
    final Pointer alpha,
    final cudnnTensorDescriptor xDesc,
    final Pointer x,
    final Pointer beta,
    final cudnnTensorDescriptor yDesc,
    final Pointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
    cudnnTransformTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnTransformTensor", result, handle, alpha, xDesc, x, beta, yDesc, y);
    return result;
  }
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static int getDevice() {
    final Integer integer = CuDNN.currentDevice.get();
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
    CuDNN.log("cudaGetDeviceCount", returnCode, deviceCount);
    deviceCount_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.handle(returnCode);
    return deviceCount[0];
  }
  
  /**
   * Gets device name.
   *
   * @param device the device
   * @return the device name
   */
  public static String getDeviceName(final int device) {
    return new String(CuDNN.getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }
  
  private volatile cudaDeviceProp deviceProperties;
  
  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  public static cudaDeviceProp getDeviceProperties(final int device) {
    return propertyCache.computeIfAbsent(device, deviceId -> {
      long startTime = System.nanoTime();
      final cudaDeviceProp deviceProp = new cudaDeviceProp();
      final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
      getDeviceProperties_execution.accept((System.nanoTime() - startTime) / 1e9);
      CuDNN.log("cudaGetDeviceProperties", result, deviceProp, device);
      return deviceProp;
    });
  }
  
  /**
   * For each.
   *
   * @param fn the fn
   */
  public static void forEach(final Consumer<? super CuDNN> fn) {
    contexts.getAll().forEach(x -> {
      x.initThread();
      fn.accept(x);
    });
  }
  
  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(final int cudaDeviceId) {
    if (cudaDeviceId < 0) throw new IllegalArgumentException("cudaDeviceId=" + cudaDeviceId);
    if (cudaDeviceId != getDevice()) {
      long startTime = System.nanoTime();
      final int result = JCuda.cudaSetDevice(cudaDeviceId);
      setDevice_execution.accept((System.nanoTime() - startTime) / 1e9);
      CuDNN.log("cudaSetDevice", result, cudaDeviceId);
      CuDNN.handle(result);
      CuDNN.currentDevice.set(cudaDeviceId);
    }
  }
  
  /**
   * Is oom boolean.
   *
   * @param t the t
   * @return the boolean
   */
  public static boolean isOom(final Throwable t) {
    if (t instanceof OutOfMemoryError) return true;
    if (t instanceof com.simiacryptus.mindseye.lang.OutOfMemoryError) return true;
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
        RecycleBin.DOUBLES.clear();
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        runtime.runFinalization();
        runtime.gc();
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
    for (PrintStream apiLog : CuDNN.apiLog) {
      final String paramString = null == args ? "" : Arrays.stream(args).map(CuDNN::renderToLog).reduce((a, b) -> a + ", " + b).orElse("");
      final String message = String.format("%.6f @ %s: %s(%s) = %s", (System.nanoTime() - CuDNN.start) / 1e9, Thread.currentThread().getName(), method, paramString, result);
      CuDNN.logThread.submit(() -> apiLog.println(message));
    }
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
    CuDNN.log("cudnnGetConvolutionNdForwardOutputDim", result, convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    CuDNN.handle(result);
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
    CuDNN.log("cudnnCreateActivationDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    newActivationDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetActivationDescriptor", result, desc, mode, reluNan, reluCeil);
    CuDNN.handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyActivationDescriptor, getDevice());
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
    CuDNN.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc,
                                                    3,
                                                    padding,
                                                    stride,
                                                    dilation,
                                                    mode,
                                                    dataType
                                                   );
    CuDNN.log("cudnnSetConvolutionNdDescriptor", result, convDesc, padding.length,
              padding,
              stride,
              dilation,
              mode,
              dataType);
    CuDNN.handle(result);
    return new CudaResource<cudnnConvolutionDescriptor>(convDesc, CuDNN::cudnnDestroyConvolutionDescriptor, getDevice()) {
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
    CuDNN.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    CuDNN.handle(result);
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
    CuDNN.log("cudnnSetConvolution2dDescriptor", result, convDesc,
              paddingY, // zero-padding height
              paddingX, // zero-padding width
              strideHeight, // vertical filter stride
              strideWidth, // horizontal filter stride
              dilationY, // upscale the input in x-direction
              dilationX, // upscale the input in y-direction
              mode,
              dataType);
    CuDNN.handle(result);
    return new CudaResource<>(convDesc, CuDNN::cudnnDestroyConvolutionDescriptor, getDevice());
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
    CuDNN.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetFilter4dDescriptor", result, filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    CuDNN.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CuDNN::cudnnDestroyFilterDescriptor, getDevice()) {
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
    CuDNN.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    newFilterDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetFilterNdDescriptor", result, filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    CuDNN.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CuDNN::cudnnDestroyFilterDescriptor, getDevice()) {
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
    CuDNN.log("cudnnCreateOpTensorDescriptor", result, opDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    newOpDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetOpTensorDescriptor", result, opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    CuDNN.handle(result);
    return new CudaResource<>(opDesc, CuDNN::cudnnDestroyOpTensorDescriptor, getDevice());
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
    CuDNN.log("cudnnCreateTensorDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetTensor4dDescriptor", result, desc, tensorLayout, dataType, batchCount, channels, height, width);
    CuDNN.handle(result);
    return new CudaResource<cudnnTensorDescriptor>(desc, CuDNN::cudnnDestroyTensorDescriptor, getDevice()) {
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
  
  private static String renderToLog(final Object obj) {
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
   * Load gpu contexts list. If the property disableCuDnn is setWeights to true, no GPUs will be recognized. This is
   * useful for testing CPU-only compatibility.
   *
   * @return the list
   */
  static List<CuDNN> loadGpuContexts() {
    if (Boolean.parseBoolean(System.getProperty("disableCuDnn", "false"))) {
      logger.warn("Disabled CuDNN");
      return Arrays.asList();
    }
    final int deviceCount = CuDNN.deviceCount();
    logger.info(String.format("Found %s devices", deviceCount));
    final List<Integer> devices = new ArrayList<>();
    for (int d = 0; d < deviceCount; d++) {
      int deviceNumber = d;
      //if(device>0) System.err.println(String.format("IGNORING Device %s - %s", device, getDeviceName(device)));
      CuDNN.withDevice(deviceNumber, () -> {
        logger.info(String.format("Device %s - %s", deviceNumber, CuDNN.getDeviceName(deviceNumber)));
        devices.add(deviceNumber);
        //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
        for (DeviceLimits limit : DeviceLimits.values()) {
          logger.info(String.format("Default Limit %s = %s", limit, limit.get()));
        }
        DeviceLimits.HeapSize.set(16 * 1024 * 1024 * 1024);
        DeviceLimits.FifoSize.set(8 * 1024 * 1024);
        for (DeviceLimits limit : DeviceLimits.values()) {
          logger.info(String.format("Configured Limit %s = %s", limit, limit.get()));
        }
      });
    }
    if (System.getProperties().containsKey("gpus")) {
      List<Integer> devices2 = Arrays.stream(System.getProperty("gpus").split(","))
                                     .map(Integer::parseInt).collect(Collectors.toList());
      devices.clear();
      devices.addAll(devices2);
    }
    logger.info(String.format("Found %s devices; using devices %s", deviceCount, devices));
    return devices.stream()
                  .map(i -> {
                    try {
                      return new CuDNN(i);
                    } catch (Throwable e) {
                      return null;
                    }
                  }).filter(x -> x != null).collect(Collectors.toList());
  }
  
  /**
   * Remove log boolean.
   *
   * @param apiLog the api log
   * @return the boolean
   */
  public static boolean removeLog(PrintStream apiLog) {
    return CuDNN.apiLog.remove(apiLog);
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
    CuDNN.log("cudnnCreateTensorDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    newTensorDescriptor_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnSetTensor4dDescriptorEx", result, desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    CuDNN.handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyTensorDescriptor, getDevice());
  }
  
  /**
   * Gets device name.
   *
   * @return the device name
   */
  public String getDeviceName() {
    return new String(getDeviceProperties().name, Charset.forName("ASCII")).trim();
  }
  
  /**
   * Reset all GPUs and Heap Memory
   */
  public static void reset() {
    synchronized (CuDNN.class) {
      cleanMemory();
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      try {
        IntStream.range(0, CuDNN.deviceCount()).forEach(deviceNumber -> CuDNN.withDevice(deviceNumber, () -> {
          logger.warn(String.format("Resetting Device %d", deviceNumber));
          CudaPtr.getGpuStats(deviceNumber).usedMemory.set(0);
          cudaDeviceReset();
        }));
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      CudaResource.gpuGeneration.incrementAndGet();
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
  }
  
  public static void withDevice(int n, Runnable action) {
    final int currentDevice = getDevice();
    try {
      action.run();
    } finally {
      if (currentDevice >= 0) setDevice(currentDevice);
      else CuDNN.currentDevice.remove();
    }
  }
  
  public static <T> T withDevice(int n, Supplier<T> action) {
    final int currentDevice = getDevice();
    try {
      return action.get();
    } finally {
      if (currentDevice >= 0) setDevice(currentDevice);
      else CuDNN.currentDevice.remove();
    }
  }
  
  /**
   * Call t.
   *
   * @param <T> the type parameter
   * @param fn  the fn
   * @return the t
   */
  public static <T> T run(final Function<CuDNN, T> fn) {
    if (contexts.getAll().isEmpty()) {
      return fn.apply(new CuDNN(-1));
    }
    else {
      CuDNN threadlocal = threadContext.get();
      if (threadlocal != null) {
        try {
          threadlocal.initThread();
          return fn.apply(threadlocal);
        } catch (final Exception e) {
          throw new RuntimeException(e);
        }
      }
      else {
        return contexts.run(exe -> {
          try {
            threadContext.set(exe);
            exe.initThread();
            return fn.apply(exe);
          } catch (final Exception e) {
            throw new RuntimeException(e);
          } finally {
            threadContext.remove();
          }
        });
      }
    }
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void apply(final Consumer<CuDNN> fn) {
    CuDNN threadlocal = threadContext.get();
    if (threadlocal != null) {
      try {
        threadlocal.initThread();
        fn.accept(threadlocal);
      } catch (final Exception e) {
        throw new RuntimeException(e);
      }
    }
    else {
      contexts.apply(exe -> {
        try {
          threadContext.set(exe);
          exe.initThread();
          fn.accept(exe);
        } catch (final Exception e) {
          throw new RuntimeException(e);
        } finally {
          threadContext.remove();
        }
      });
    }
  }
  
  /**
   * Is enabled boolean.
   *
   * @return the boolean
   */
  public static boolean isEnabled() {
    return 0 < CuDNN.contexts.size();
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
   * Allocate backward data workspace cuda ptr.
   *
   * @param deviceId   the device id
   * @param inputDesc  the input desc
   * @param filterDesc the filter desc
   * @param convDesc   the conv desc
   * @param outputDesc the output desc
   * @param algorithm  the algorithm
   * @return the cuda ptr
   */
  public CudaPtr allocateBackwardDataWorkspace(final int deviceId, final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                                           filterDesc, outputDesc, convDesc, inputDesc,
                                                                           algorithm, sizeInBytesArray);
    allocateBackwardDataWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionBackwardDataWorkspaceSize", result, cudnnHandle,
              filterDesc, outputDesc, convDesc, inputDesc,
              algorithm, sizeInBytesArray);
    CuDNN.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    return CuDNN.alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
  }
  
  /**
   * Allocate backward filter workspace cuda ptr.
   *
   * @param deviceId      the device id
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public CudaPtr allocateBackwardFilterWorkspace(final int deviceId, final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                             srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                                                                             algorithm, sizeInBytesArray);
    allocateBackwardFilterWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionBackwardFilterWorkspaceSize", result, cudnnHandle,
              srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
              algorithm, sizeInBytesArray);
    CuDNN.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    return CuDNN.alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
  }
  
  /**
   * Allocate forward workspace cuda ptr.
   *
   * @param deviceId      the device id
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public CudaPtr allocateForwardWorkspace(final int deviceId, final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                      algorithm, sizeInBytesArray);
    allocateForwardWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionForwardWorkspaceSize", result, cudnnHandle,
              srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
              algorithm, sizeInBytesArray);
    CuDNN.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    return CuDNN.alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0, true);
  }
  
  @Override
  public void finalize() throws Throwable {
    final int result = JCudnn.cudnnDestroy(cudnnHandle);
    CuDNN.log("cudnnDestroy", result, cudnnHandle);
    CuDNN.handle(result);
  }
  
  /**
   * Gets backward data algorithm.
   *
   * @param inputDesc  the src tensor desc
   * @param filterDesc the filter desc
   * @param convDesc   the conv desc
   * @param outputDesc the weight desc
   * @return the backward data algorithm
   */
  public int getBackwardDataAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                                                                       filterDesc, inputDesc, convDesc, outputDesc,
                                                                       cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardDataAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionBackwardDataAlgorithm", result, cudnnHandle,
              filterDesc, inputDesc, convDesc, outputDesc,
              cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    CuDNN.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets backward filter algorithm.
   *
   * @param inputDesc  the input desc
   * @param filterDesc the filter desc
   * @param convDesc   the conv desc
   * @param outputDesc the output desc
   * @return the backward filter algorithm
   */
  public int getBackwardFilterAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                                                                         inputDesc, outputDesc, convDesc, filterDesc,
                                                                         cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardFilterAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionBackwardFilterAlgorithm", result, cudnnHandle,
              inputDesc, outputDesc, convDesc, filterDesc,
              cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    CuDNN.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets device number.
   *
   * @return the device number
   */
  public int getDeviceNumber() {
    return deviceNumber;
  }
  
  /**
   * Gets forward algorithm.
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @return the forward algorithm
   */
  public int getForwardAlgorithm(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                                  srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                  cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getForwardAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    CuDNN.log("cudnnGetConvolutionForwardAlgorithm", result, cudnnHandle,
              srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
              cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    CuDNN.handle(result);
    return algoArray[0];
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    CuDNN.setDevice(getDeviceNumber());
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceNumber + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
  
  /**
   * Gets device properties.
   *
   * @return the device properties
   */
  public cudaDeviceProp getDeviceProperties() {
    if (null == deviceProperties) {
      synchronized (this) {
        if (null == deviceProperties) {
          deviceProperties = getDeviceProperties(getDeviceNumber());
        }
      }
    }
    return deviceProperties;
  }
}