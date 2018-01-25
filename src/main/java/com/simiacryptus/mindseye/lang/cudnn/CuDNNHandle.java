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

import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.util.lang.StaticResourcePool;
import jcuda.Pointer;
import jcuda.jcudnn.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Gpu handle.
 */
public class CuDNNHandle extends GpuDevice {
  public static final ThreadLocal<LinkedBlockingDeque<ReferenceCounting>> CLEANUP = new ThreadLocal<LinkedBlockingDeque<ReferenceCounting>>() {
    @Override
    protected LinkedBlockingDeque<ReferenceCounting> initialValue() {
      return new LinkedBlockingDeque<>();
    }
  };
  /**
   * The constant gpuContexts.
   */
  private static final boolean DISABLE = Boolean.parseBoolean(System.getProperty("DISABLE_CUDNN", Boolean.toString(false)));
  private static final boolean FORCE_SINGLE_GPU = Boolean.parseBoolean(System.getProperty("FORCE_SINGLE_GPU", Boolean.toString(false)));
  private static final int THREADS_PER_GPU = Integer.parseInt(System.getProperty("THREADS_PER_GPU", Integer.toString(3)));
  public static final StaticResourcePool<CuDNNHandle> POOL = new StaticResourcePool<>(loadGpuContexts());
  private static final ThreadLocal<CuDNNHandle> threadContext = new ThreadLocal<>();
  private final jcuda.jcudnn.cudnnHandle handle;
  
  /**
   * Instantiates a new Cu dnn.
   *
   * @param deviceNumber the device number
   */
  private CuDNNHandle(final int deviceNumber) {
    super(deviceNumber);
    if (0 <= this.deviceNumber) {
      initThread();
      handle = new cudnnHandle();
      JCudnn.cudnnCreate(getHandle());
    }
    else {
      handle = null;
    }
    //cudaSetDevice();
  }
  
  /**
   * Run.
   *
   * @param fn the fn
   */
  public static void apply(final Consumer<CuDNNHandle> fn) {apply(fn, true);}
  
  /**
   * Run.
   *
   * @param fn          the fn
   * @param synchronize the synchronize
   */
  public static void apply(final Consumer<CuDNNHandle> fn, boolean synchronize) {
    CuDNNHandle threadlocal = threadContext.get();
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
        POOL.apply(exe -> {
          try {
            threadContext.set(exe);
            exe.initThread();
            fn.accept(exe);
          } catch (final RuntimeException e) {
            throw e;
          } catch (final Exception e) {
            throw new RuntimeException(e);
          } finally {
            threadContext.remove();
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
  public static <T> T run(final Function<CuDNNHandle, T> fn) {return run(fn, true);}
  
  /**
   * Call t.
   *
   * @param <T>         the type parameter
   * @param fn          the fn
   * @param synchronize the synchronize
   * @return the t
   */
  public static <T> T run(final Function<CuDNNHandle, T> fn, boolean synchronize) {
    if (POOL.getAll().isEmpty()) {
      return fn.apply(new CuDNNHandle(-1));
    }
    else {
      try {
        CuDNNHandle threadlocal = threadContext.get();
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
          return POOL.run(exe -> {
            try {
              threadContext.set(exe);
              exe.initThread();
              T result = fn.apply(exe);
              return result;
            } catch (final RuntimeException e) {
              throw e;
            } catch (final Exception e) {
              throw new RuntimeException(e);
            } finally {
              threadContext.remove();
            }
          });
        }
      } finally {
        if (synchronize) GpuSystem.cudaDeviceSynchronize();
        LinkedBlockingDeque<ReferenceCounting> deque = CLEANUP.get();
        deque.stream().forEach(x -> x.freeRef());
        deque.clear();
      }
    }
  }
  
  /**
   * Load gpu contexts list. If the property disableCuDnn is set to true, no GPUs will be recognized. This is useful for
   * testing CPU-only compatibility.
   *
   * @return the list
   */
  private static List<CuDNNHandle> loadGpuContexts() {
    if (DISABLE) {
      logger.warn("Disabled GpuSystem");
      return Arrays.asList();
    }
    final int deviceCount;
    if (FORCE_SINGLE_GPU) {
      logger.warn("Forcing Single-GPU Mode");
      deviceCount = 1;
    }
    else {
      deviceCount = GpuSystem.deviceCount();
    }
    logger.info(String.format("Found %s devices", deviceCount));
    final List<Integer> devices = new ArrayList<>();
    for (int d = 0; d < deviceCount; d++) {
      int deviceNumber = d;
      //if(device>0) System.err.println(String.format("IGNORING Device %s - %s", device, getDeviceName(device)));
      GpuSystem.withDevice(deviceNumber, () -> {
        logger.info(String.format("Device %s - %s", deviceNumber, GpuDevice.getDeviceName(deviceNumber)));
        devices.add(deviceNumber);
        try {
          //GpuSystem.handle(GpuSystem.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync));
        } catch (Throwable e) {
          logger.warn("Error initializing GPU", e);
          throw new RuntimeException(e);
        }
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
    List<CuDNNHandle> handles = devices.stream()
                                       .flatMap(i -> {
                                         try {
                                           return IntStream.range(0, THREADS_PER_GPU).mapToObj(j -> new CuDNNHandle(i));
                                         } catch (Throwable e) {
                                           logger.warn(String.format("Error initializing device %d", i), e);
                                           return Stream.empty();
                                         }
                                       }).collect(Collectors.toList());
    logger.info(String.format("Found %s devices; using %s handles per devices %s; %s handles", deviceCount, THREADS_PER_GPU, devices, handles.size()));
    return handles;
  }
  
  /**
   * For each.
   *
   * @param fn the fn
   */
  public static void forEach(final Consumer<? super GpuDevice> fn) {
    POOL.getAll().forEach(x -> {
      x.initThread();
      fn.accept(x);
    });
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
    GpuSystem.log("cudnnActivationBackward", result, handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    GpuSystem.log("cudnnActivationForward", result, handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
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
    GpuSystem.log("cudnnAddTensor", result, handle, alpha, aDesc, A, beta, cDesc, C);
    GpuSystem.handle(result);
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
    GpuSystem.log("cudnnConvolutionBackwardBias", result, handle, alpha, dyDesc, dy, beta, dbDesc, db);
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
    GpuSystem.log("cudnnConvolutionBackwardData", result, handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
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
    GpuSystem.log("cudnnConvolutionBackwardFilter", result, handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
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
    GpuSystem.log("cudnnConvolutionForward", result, handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
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
    GpuSystem.log("cudnnOpTensor", result, handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
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
    GpuSystem.log("cudnnPoolingBackward", result, handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    GpuSystem.log("cudnnPoolingForward", result, handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
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
    GpuSystem.log("cudnnTransformTensor", result, handle, alpha, xDesc, x, beta, yDesc, y);
    return result;
  }
  
  /**
   * Allocate backward data workspace cuda ptr.
   *
   * @param cudnnHandle the cudnn handle
   * @param deviceId    the device id
   * @param inputDesc   the input desc
   * @param filterDesc  the filter desc
   * @param convDesc    the conv desc
   * @param outputDesc  the output desc
   * @param algorithm   the algorithm
   * @return the cuda ptr
   */
  public static CudaPtr allocateBackwardDataWorkspace(cudnnHandle cudnnHandle, final int deviceId, final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                                           filterDesc, outputDesc, convDesc, inputDesc,
                                                                           algorithm, sizeInBytesArray);
    allocateBackwardDataWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionBackwardDataWorkspaceSize", result, cudnnHandle,
                  filterDesc, outputDesc, convDesc, inputDesc,
                  algorithm, sizeInBytesArray);
    GpuSystem.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    final long size = 0 < workspaceSize ? workspaceSize : 0;
    return CudaPtr.allocate(deviceId, size, MemoryType.Managed, true);
  }
  
  /**
   * Allocate backward filter workspace cuda ptr.
   *
   * @param cudnnHandle   the cudnn handle
   * @param deviceId      the device id
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public static CudaPtr allocateBackwardFilterWorkspace(cudnnHandle cudnnHandle, final int deviceId, final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                             srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                                                                             algorithm, sizeInBytesArray);
    allocateBackwardFilterWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionBackwardFilterWorkspaceSize", result, cudnnHandle,
                  srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                  algorithm, sizeInBytesArray);
    GpuSystem.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    final long size = 0 < workspaceSize ? workspaceSize : 0;
    return CudaPtr.allocate(deviceId, size, MemoryType.Managed, true);
  }
  
  /**
   * Allocate forward workspace cuda ptr.
   *
   * @param cudnnHandle   the cudnn handle
   * @param deviceId      the device id
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public static CudaPtr allocateForwardWorkspace(cudnnHandle cudnnHandle, final int deviceId, final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                      algorithm, sizeInBytesArray);
    allocateForwardWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionForwardWorkspaceSize", result, cudnnHandle,
                  srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                  algorithm, sizeInBytesArray);
    GpuSystem.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    final long size = 0 < workspaceSize ? workspaceSize : 0;
    return CudaPtr.allocate(deviceId, size, MemoryType.Managed, true);
  }
  
  /**
   * Gets backward data algorithm.
   *
   * @param cudnnHandle the cudnn handle
   * @param inputDesc   the src tensor desc
   * @param filterDesc  the filter desc
   * @param convDesc    the conv desc
   * @param outputDesc  the weight desc
   * @return the backward data algorithm
   */
  public static int getBackwardDataAlgorithm(cudnnHandle cudnnHandle, final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                                                                       filterDesc, inputDesc, convDesc, outputDesc,
                                                                       cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardDataAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionBackwardDataAlgorithm", result, cudnnHandle,
                  filterDesc, inputDesc, convDesc, outputDesc,
                  cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    GpuSystem.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets backward filter algorithm.
   *
   * @param cudnnHandle the cudnn handle
   * @param inputDesc   the input desc
   * @param filterDesc  the filter desc
   * @param convDesc    the conv desc
   * @param outputDesc  the output desc
   * @return the backward filter algorithm
   */
  public static int getBackwardFilterAlgorithm(cudnnHandle cudnnHandle, final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                                                                         inputDesc, outputDesc, convDesc, filterDesc,
                                                                         cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardFilterAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionBackwardFilterAlgorithm", result, cudnnHandle,
                  inputDesc, outputDesc, convDesc, filterDesc,
                  cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    GpuSystem.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets forward algorithm.
   *
   * @param cudnnHandle   the cudnn handle
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @return the forward algorithm
   */
  public static int getForwardAlgorithm(cudnnHandle cudnnHandle, final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc) {
    long startTime = System.nanoTime();
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                                  srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                  cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getForwardAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    GpuSystem.log("cudnnGetConvolutionForwardAlgorithm", result, cudnnHandle,
                  srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                  cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    GpuSystem.handle(result);
    return algoArray[0];
  }
  
  public void registerForCleanup(ReferenceCounting... objs) {
    Arrays.stream(objs).forEach(ReferenceCounting::assertAlive);
    LinkedBlockingDeque<ReferenceCounting> list = CLEANUP.get();
    Arrays.stream(objs).forEach(list::add);
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceNumber + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
  
  @Override
  public void finalize() throws Throwable {
    final int result = JCudnn.cudnnDestroy(getHandle());
    GpuSystem.log("cudnnDestroy", result, getHandle());
    GpuSystem.handle(result);
  }
  
  /**
   * The Cudnn handle.
   *
   * @return the handle
   */
  public cudnnHandle getHandle() {
    return handle;
  }
}
