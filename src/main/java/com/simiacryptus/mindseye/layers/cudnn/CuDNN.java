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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.mindseye.lang.GpuError;
import jcuda.Pointer;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

/**
 * Main library wrapper class around the CuDNN API, providing logging and managed wrappers.
 */
public class CuDNN {
  
  private static final ThreadLocal<Integer> currentDevice = new ThreadLocal<Integer>() {
    @Override
    protected Integer initialValue() {
      return -1;
    }
  };
  private static final ExecutorService logThread = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder().setDaemon(true).build());
  private static final long start = System.nanoTime();
  /**
   * The constant apiLog.
   */
  public static PrintStream apiLog = null;
  /**
   * The Cudnn handle.
   */
  public final cudnnHandle cudnnHandle;
  private final String deviceName;
  private final int deviceNumber;
  
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
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @return the cuda ptr
   */
  public static CudaPtr alloc(final int deviceId, final long size) {
    return new CudaPtr(size, deviceId);
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
    return new CudaResource<>(poolingDesc, CuDNN::cudnnDestroyPoolingDescriptor);
  }
  
  /**
   * Cuda device reset int.
   *
   * @return the int
   */
  public static int cudaDeviceReset() {
    final int result = JCuda.cudaDeviceReset();
    CuDNN.log("cudaDeviceReset", result);
    return result;
  }
  
  /**
   * Cuda finalize int.
   *
   * @param devPtr the dev ptr
   * @return the int
   */
  public static int cudaFree(final Pointer devPtr) {
    final int result = JCuda.cudaFree(devPtr);
    CuDNN.log("cudaFree", result, devPtr);
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
    final int result = JCuda.cudaMalloc(devPtr, size);
    CuDNN.log("cudaMalloc", result, devPtr, size);
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
    final int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    CuDNN.log("cudaMemcpy", result, dst, src, count, cudaMemcpyKind_kind);
    return result;
  }
  
  /**
   * Cuda memset int.
   *
   * @param mem   the mem
   * @param c     the c
   * @param count the count
   * @return the int
   */
  public static int cudaMemset(final Pointer mem, final int c, final long count) {
    final int result = JCuda.cudaMemset(mem, c, count);
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
    final int result = JCudnn.cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    final int result = JCudnn.cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
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
    final int result = JCudnn.cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
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
    final int result = JCudnn.cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
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
    final int result = JCudnn.cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
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
    final int result = JCudnn.cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
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
    final int result = JCudnn.cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    CuDNN.log("cudnnConvolutionForward", result, handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    return result;
  }

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
  
  /**
   * Cudnn destroy activation descriptor int.
   *
   * @param activationDesc the activation desc
   * @return the int
   */
  public static int cudnnDestroyActivationDescriptor(final cudnnActivationDescriptor activationDesc) {
    final int result = JCudnn.cudnnDestroyActivationDescriptor(activationDesc);
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
    final int result = JCudnn.cudnnDestroyConvolutionDescriptor(convDesc);
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
    final int result = JCudnn.cudnnDestroyFilterDescriptor(filterDesc);
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
    final int result = JCudnn.cudnnDestroyOpTensorDescriptor(opTensorDesc);
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
    final int result = JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
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
    final int result = JCudnn.cudnnDestroyTensorDescriptor(tensorDesc);
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
    final int result = JCudnn.cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
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
    final int result = JCudnn.cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
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
    final int result = JCudnn.cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    final int result = JCudnn.cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    CuDNN.log("cudnnPoolingForward", result, handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
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
    final int result = JCudnn.cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
    CuDNN.log("cudnnTransformTensor", result, handle, alpha, xDesc, x, beta, yDesc, y);
    return result;
  }
  
  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    final int[] deviceCount = new int[1];
    final int returnCode = jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    CuDNN.log("cudaGetDeviceCount", returnCode, deviceCount);
    CuDNN.handle(returnCode);
    return deviceCount[0];
  }
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static int getDevice() {
    final Integer integer = CuDNN.currentDevice.get();
    return integer == null ? 0 : integer;
  }
  
  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(final int cudaDeviceId) {
    final int result = JCuda.cudaSetDevice(cudaDeviceId);
    CuDNN.log("cudaSetDevice", result, cudaDeviceId);
    CuDNN.handle(result);
    CuDNN.currentDevice.set(cudaDeviceId);
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
  
  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  public static cudaDeviceProp getDeviceProperties(final int device) {
    final cudaDeviceProp deviceProp = new cudaDeviceProp();
    final int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
    CuDNN.log("cudaGetDeviceProperties", result, deviceProp, device);
    return deviceProp;
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
    final int[] tensorOuputDims = new int[4];
    final int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    CuDNN.log("cudnnGetConvolutionNdForwardOutputDim", result, convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    CuDNN.handle(result);
    return tensorOuputDims;
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
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  protected static void log(final String method, final Object result, final Object... args) {
    final PrintStream apiLog = CuDNN.apiLog;
    if (null != apiLog) {
      final String paramString = Arrays.stream(args).map(CuDNN::renderToLog).reduce((a, b) -> a + ", " + b).get();
      final String message = String.format("%.6f: %s(%s) = %s", (System.nanoTime() - CuDNN.start) / 1e9, method, paramString, result);
      CuDNN.logThread.submit(() -> apiLog.println(message));
    }
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
    final cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    CuDNN.log("cudnnCreateActivationDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    CuDNN.log("cudnnSetActivationDescriptor", result, desc, mode, reluNan, reluCeil);
    CuDNN.handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyActivationDescriptor);
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
    final cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    CuDNN.log("cudnnCreateConvolutionDescriptor", result, convDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc, padding.length,
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
    return new CudaResource<cudnnConvolutionDescriptor>(convDesc, CuDNN::cudnnDestroyConvolutionDescriptor) {
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
   * @param paddingX     the padding x
   * @param paddingY     the padding y
   * @param strideHeight the stride height
   * @param strideWidth  the stride width
   * @param mode         the mode
   * @param dataType     the data type
   * @return the cuda resource
   */
  public static CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(final int paddingX, final int paddingY, final int strideHeight, final int strideWidth, final int mode, final int dataType) {
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
      1, // upscale the input in x-direction
      1, // upscale the input in y-direction
      mode
      , dataType
                                                   );
    CuDNN.log("cudnnSetConvolution2dDescriptor", result, convDesc,
              paddingY, // zero-padding height
              paddingX, // zero-padding width
              strideHeight, // vertical filter stride
              strideWidth, // horizontal filter stride
              1, // upscale the input in x-direction
              1, // upscale the input in y-direction
              mode,
              dataType);
    CuDNN.handle(result);
    return new CudaResource<>(convDesc, CuDNN::cudnnDestroyConvolutionDescriptor);
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
    final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    CuDNN.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    CuDNN.log("cudnnSetFilter4dDescriptor", result, filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    CuDNN.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CuDNN::cudnnDestroyFilterDescriptor) {
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
    final cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    CuDNN.log("cudnnCreateFilterDescriptor", result, filterDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    CuDNN.log("cudnnSetFilterNdDescriptor", result, filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    CuDNN.handle(result);
    return new CudaResource<cudnnFilterDescriptor>(filterDesc, CuDNN::cudnnDestroyFilterDescriptor) {
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
    final cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    CuDNN.log("cudnnCreateOpTensorDescriptor", result, opDesc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    CuDNN.log("cudnnSetOpTensorDescriptor", result, opDesc, opType, dataType, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN);
    CuDNN.handle(result);
    
    return new CudaResource<>(opDesc, CuDNN::cudnnDestroyOpTensorDescriptor);
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
    final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    CuDNN.log("cudnnCreateTensorDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width);
    CuDNN.log("cudnnSetTensor4dDescriptor", result, desc, tensorLayout, dataType, batchCount, channels, height, width);
    CuDNN.handle(result);
    return new CudaResource<cudnnTensorDescriptor>(desc, CuDNN::cudnnDestroyTensorDescriptor) {
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
    final cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    CuDNN.log("cudnnCreateTensorDescriptor", result, desc);
    CuDNN.handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    CuDNN.log("cudnnSetTensor4dDescriptorEx", result, desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    CuDNN.handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyTensorDescriptor);
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
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                                           filterDesc, outputDesc, convDesc, inputDesc,
                                                                           algorithm, sizeInBytesArray);
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
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                             srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                                                                             algorithm, sizeInBytesArray);
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
    final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                      algorithm, sizeInBytesArray);
    CuDNN.log("cudnnGetConvolutionForwardWorkspaceSize", result, cudnnHandle,
              srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
              algorithm, sizeInBytesArray);
    CuDNN.handle(result);
    final long workspaceSize = sizeInBytesArray[0];
    return CuDNN.alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
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
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param weightDesc    the weight desc
   * @return the backward data algorithm
   */
  public int getBackwardDataAlgorithm(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor weightDesc) {
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                                                                       filterDesc, srcTensorDesc, convDesc, weightDesc,
                                                                       cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    CuDNN.log("cudnnGetConvolutionBackwardDataAlgorithm", result, cudnnHandle,
              filterDesc, srcTensorDesc, convDesc, weightDesc,
              cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
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
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                                                                         inputDesc, outputDesc, convDesc, filterDesc,
                                                                         cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    CuDNN.log("cudnnGetConvolutionBackwardFilterAlgorithm", result, cudnnHandle,
              inputDesc, outputDesc, convDesc, filterDesc,
              cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
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
    final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                                  srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                                                                  cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    CuDNN.log("cudnnGetConvolutionForwardAlgorithm", result, cudnnHandle,
              srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
              cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    CuDNN.handle(result);
    return algoArray[0];
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    CuDNN.setDevice(getDeviceNumber());
    //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceNumber + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
}
