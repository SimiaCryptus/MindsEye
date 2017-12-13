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

import static jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS;

/**
 * The type Cu dnn.
 */
public class CuDNN {
  
  private static final ThreadLocal<Integer> currentDevice = new ThreadLocal<Integer>() {
    @Override
    protected Integer initialValue() {
      return -1;
    }
  };
  private static final long start = System.nanoTime();
  private static final ExecutorService logThread = Executors.newSingleThreadExecutor();
  /**
   * The constant apiLog.
   */
  public static PrintStream apiLog = null;
  /**
   * The Cudnn handle.
   */
  public final cudnnHandle cudnnHandle;
  private final int deviceNumber;
  private final String deviceName;
  
  /**
   * Instantiates a new Cu dnn.
   *
   * @param deviceNumber the device number
   */
  protected CuDNN(int deviceNumber) {
    this.deviceNumber = deviceNumber;
    this.cudnnHandle = new cudnnHandle();
    initThread();
    this.deviceName = getDeviceName(deviceNumber);
    JCudnn.cudnnCreate(cudnnHandle);
    //cudaSetDevice();
  }
  
  /**
   * Device count int.
   *
   * @return the int
   */
  public static int deviceCount() {
    int[] deviceCount = new int[1];
    int returnCode = jcuda.runtime.JCuda.cudaGetDeviceCount(deviceCount);
    log("cudaGetDeviceCount", returnCode, deviceCount);
    handle(returnCode);
    return deviceCount[0];
  }
  
  /**
   * Log.
   *
   * @param method the method
   * @param result the result
   * @param args   the args
   */
  protected static void log(String method, Object result, Object... args) {
    PrintStream apiLog = CuDNN.apiLog;
    if (null != apiLog) {
      String paramString = Arrays.stream(args).map(CuDNN::renderToLog).reduce((a, b) -> a + ", " + b).get();
      String message = String.format("%.6f: %s(%s) = %s", (System.nanoTime() - start) / 1e9, method, paramString, result);
      logThread.submit(() -> apiLog.println(message));
    }
  }
  
  private static String renderToLog(Object obj) {
    if (obj instanceof int[]) {
      if (((int[]) obj).length < 10) {
        return Arrays.toString(((int[]) obj));
      }
    }
    if (obj instanceof double[]) {
      if (((double[]) obj).length < 10) {
        return Arrays.toString(((double[]) obj));
      }
    }
    if (obj instanceof float[]) {
      if (((float[]) obj).length < 10) {
        return Arrays.toString(((float[]) obj));
      }
    }
    if (obj instanceof long[]) {
      if (((long[]) obj).length < 10) {
        return Arrays.toString(((long[]) obj));
      }
    }
    return obj.toString();
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
  public static CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(int mode, int poolDims, int[] windowSize, int[] padding, int[] stride) {
    cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
    int result = JCudnn.cudnnCreatePoolingDescriptor(poolingDesc);
    log("cudnnCreatePoolingDescriptor", result, poolingDesc);
    handle(result);
    result = JCudnn.cudnnSetPoolingNdDescriptor(poolingDesc,
      mode, CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
      padding, stride);
    log("cudnnSetPoolingNdDescriptor", result, poolingDesc,
      mode, CUDNN_NOT_PROPAGATE_NAN, poolDims, windowSize,
      padding, stride);
    handle(result);
    return new CudaResource<cudnnPoolingDescriptor>(poolingDesc, CuDNN::cudnnDestroyPoolingDescriptor);
  }
  
  /**
   * Cudnn destroy pooling descriptor int.
   *
   * @param poolingDesc the pooling desc
   * @return the int
   */
  public static int cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor poolingDesc) {
    int result = JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
    log("cudnnDestroyPoolingDescriptor", result, poolingDesc);
    return result;
  }
  
  /**
   * Get output dims int [ ].
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @return the int [ ]
   */
  public static int[] getOutputDims(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc) {
    int[] tensorOuputDims = new int[4];
    int result = JCudnn.cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    log("cudnnGetConvolutionNdForwardOutputDim", result, convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims);
    handle(result);
    return tensorOuputDims;
  }
  
  /**
   * Handle.
   *
   * @param returnCode the return code
   */
  public static void handle(int returnCode) {
    if (returnCode != CUDNN_STATUS_SUCCESS) {
      throw new GpuError("returnCode = " + cudnnStatus.stringFor(returnCode));
    }
  }
  
  /**
   * Gets device name.
   *
   * @param device the device
   * @return the device name
   */
  public static String getDeviceName(int device) {
    return new String(getDeviceProperties(device).name, Charset.forName("ASCII")).trim();
  }
  
  /**
   * Gets device properties.
   *
   * @param device the device
   * @return the device properties
   */
  public static cudaDeviceProp getDeviceProperties(int device) {
    cudaDeviceProp deviceProp = new cudaDeviceProp();
    int result = JCuda.cudaGetDeviceProperties(deviceProp, device);
    log("cudaGetDeviceProperties", result, deviceProp, device);
    return deviceProp;
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
  public static CudaResource<cudnnConvolutionDescriptor> newConvolutionNdDescriptor(int mode, int dataType, int[] padding, int[] stride, int[] dilation) {
    cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    log("cudnnCreateConvolutionDescriptor", result, convDesc);
    handle(result);
    result = JCudnn.cudnnSetConvolutionNdDescriptor(convDesc, padding.length,
      padding,
      stride,
      dilation,
      mode,
      dataType
    );
    log("cudnnSetConvolutionNdDescriptor", result, convDesc, padding.length,
      padding,
      stride,
      dilation,
      mode,
      dataType);
    handle(result);
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
   * Cudnn destroy convolution descriptor int.
   *
   * @param convDesc the conv desc
   * @return the int
   */
  public static int cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor convDesc) {
    int result = JCudnn.cudnnDestroyConvolutionDescriptor(convDesc);
    log("cudnnDestroyConvolutionDescriptor", result, convDesc);
    return result;
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
  public static CudaResource<cudnnConvolutionDescriptor> newConvolutions2dDescriptor(int paddingX, int paddingY, int strideHeight, int strideWidth, int mode, int dataType) {
    cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
    int result = JCudnn.cudnnCreateConvolutionDescriptor(convDesc);
    log("cudnnCreateConvolutionDescriptor", result, convDesc);
    handle(result);
    result = JCudnn.cudnnSetConvolution2dDescriptor(
      convDesc,
      paddingY, // zero-padding height
      paddingX, // zero-padding width
      strideHeight, // vertical filter stride
      strideWidth, // horizontal filter stride
      1, // upscale the input in x-direction
      1, // upscale the input in y-direction
      mode,
      dataType
    );
    log("cudnnSetConvolution2dDescriptor", result, convDesc,
      paddingY, // zero-padding height
      paddingX, // zero-padding width
      strideHeight, // vertical filter stride
      strideWidth, // horizontal filter stride
      1, // upscale the input in x-direction
      1, // upscale the input in y-direction
      mode,
      dataType);
    handle(result);
    return new CudaResource<>(convDesc, CuDNN::cudnnDestroyConvolutionDescriptor);
  }
  
  /**
   * New filter descriptor cuda resource.
   *
   * @param dataType     the data type
   * @param tensorLayout the tensor layout
   * @param dimensions   the dimensions
   * @return the cuda resource
   */
  public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int[] dimensions) {
    cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, filterDesc);
    handle(result);
    result = JCudnn.cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    log("cudnnSetFilterNdDescriptor", result, filterDesc, dataType, tensorLayout, dimensions.length, dimensions);
    handle(result);
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
   * Get stride int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  public static int[] getStride(int[] array) {
    return IntStream.range(0, array.length).map(i -> IntStream.range(i + 1, array.length).map(ii -> array[ii]).reduce((a, b) -> a * b).orElse(1)).toArray();
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
  public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int outputChannels, int inputChannels, int height, int width) {
    cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
    int result = JCudnn.cudnnCreateFilterDescriptor(filterDesc);
    log("cudnnCreateFilterDescriptor", result, filterDesc);
    handle(result);
    result = JCudnn.cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    log("cudnnSetFilter4dDescriptor", result, filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width);
    handle(result);
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
   * Cudnn destroy filter descriptor int.
   *
   * @param filterDesc the filter desc
   * @return the int
   */
  public static int cudnnDestroyFilterDescriptor(cudnnFilterDescriptor filterDesc) {
    int result = JCudnn.cudnnDestroyFilterDescriptor(filterDesc);
    log("cudnnDestroyFilterDescriptor", result, filterDesc);
    return result;
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
  public static CudaResource<cudnnTensorDescriptor> newTensorDescriptor(int dataType, int tensorLayout,
                                                                        int batchCount, int channels, int height, int width) {
    cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    log("cudnnCreateTensorDescriptor", result, desc);
    handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width);
    log("cudnnSetTensor4dDescriptor", result, desc, tensorLayout, dataType, batchCount, channels, height, width);
    handle(result);
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
   * Cudnn destroy tensor descriptor int.
   *
   * @param tensorDesc the tensor desc
   * @return the int
   */
  public static int cudnnDestroyTensorDescriptor(cudnnTensorDescriptor tensorDesc) {
    int result = JCudnn.cudnnDestroyTensorDescriptor(tensorDesc);
    log("cudnnDestroyTensorDescriptor", result, tensorDesc);
    return result;
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
  public static CudaResource<cudnnTensorDescriptor> newTensorDescriptor(int dataType,
                                                                        int batchCount, int channels, int height, int width,
                                                                        int nStride, int cStride, int hStride, int wStride) {
    cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
    int result = JCudnn.cudnnCreateTensorDescriptor(desc);
    log("cudnnCreateTensorDescriptor", result, desc);
    handle(result);
    result = JCudnn.cudnnSetTensor4dDescriptorEx(desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    log("cudnnSetTensor4dDescriptorEx", result, desc, dataType, batchCount, channels, height, width, nStride, cStride, hStride, wStride);
    handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyTensorDescriptor);
  }
  
  /**
   * New activation descriptor cuda resource.
   *
   * @param mode     the mode
   * @param reluNan  the relu nan
   * @param reluCeil the relu ceil
   * @return the cuda resource
   */
  public static CudaResource<cudnnActivationDescriptor> newActivationDescriptor(int mode, int reluNan, double reluCeil) {
    cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
    int result = JCudnn.cudnnCreateActivationDescriptor(desc);
    log("cudnnCreateActivationDescriptor", result, desc);
    handle(result);
    result = JCudnn.cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil);
    log("cudnnSetActivationDescriptor", result, desc, mode, reluNan, reluCeil);
    handle(result);
    return new CudaResource<>(desc, CuDNN::cudnnDestroyActivationDescriptor);
  }
  
  /**
   * Cudnn destroy activation descriptor int.
   *
   * @param activationDesc the activation desc
   * @return the int
   */
  public static int cudnnDestroyActivationDescriptor(cudnnActivationDescriptor activationDesc) {
    int result = JCudnn.cudnnDestroyActivationDescriptor(activationDesc);
    log("cudnnDestroyActivationDescriptor", result, activationDesc);
    return result;
  }
  
  /**
   * Alloc cuda ptr.
   *
   * @param deviceId the device id
   * @param size     the size
   * @return the cuda ptr
   */
  public static CudaPtr alloc(int deviceId, long size) {
    return new CudaPtr(size, deviceId);
  }
  
  /**
   * Gets device.
   *
   * @return the device
   */
  public static int getDevice() {
    Integer integer = currentDevice.get();
    return integer == null ? 0 : integer;
  }
  
  /**
   * Sets device.
   *
   * @param cudaDeviceId the cuda device id
   */
  public static void setDevice(int cudaDeviceId) {
    int result = JCuda.cudaSetDevice(cudaDeviceId);
    log("cudaSetDevice", result, cudaDeviceId);
    handle(result);
    currentDevice.set(cudaDeviceId);
  }
  
  /**
   * New op descriptor cuda resource.
   *
   * @param opType   the op type
   * @param dataType the data type
   * @return the cuda resource
   */
  public static CudaResource<cudnnOpTensorDescriptor> newOpDescriptor(int opType, int dataType) {
    cudnnOpTensorDescriptor opDesc = new cudnnOpTensorDescriptor();
    int result = JCudnn.cudnnCreateOpTensorDescriptor(opDesc);
    log("cudnnCreateOpTensorDescriptor", result, opDesc);
    handle(result);
    result = JCudnn.cudnnSetOpTensorDescriptor(opDesc, opType, dataType, CUDNN_NOT_PROPAGATE_NAN);
    log("cudnnSetOpTensorDescriptor", result, opDesc, opType, dataType, CUDNN_NOT_PROPAGATE_NAN);
    handle(result);
    
    return new CudaResource<>(opDesc, CuDNN::cudnnDestroyOpTensorDescriptor);
  }
  public static int cudnnOpTensor(
    cudnnHandle handle,
    cudnnOpTensorDescriptor opTensorDesc,
    Pointer alpha1,
    cudnnTensorDescriptor aDesc,
    Pointer A,
    Pointer alpha2,
    cudnnTensorDescriptor bDesc,
    Pointer B,
    Pointer beta,
    cudnnTensorDescriptor cDesc,
    Pointer C)
  {
    int result = JCudnn.cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    log("cudnnOpTensor", result, handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    return result;
  }
  
  /**
   * Cudnn destroy op tensor descriptor int.
   *
   * @param opTensorDesc the op tensor desc
   * @return the int
   */
  public static int cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor opTensorDesc) {
    int result = JCudnn.cudnnDestroyOpTensorDescriptor(opTensorDesc);
    log("cudnnDestroyOpTensorDescriptor", result, opTensorDesc);
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
    cudnnHandle handle,
    cudnnActivationDescriptor activationDesc,
    Pointer alpha,
    cudnnTensorDescriptor yDesc,
    Pointer y,
    cudnnTensorDescriptor dyDesc,
    Pointer dy,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    Pointer beta,
    cudnnTensorDescriptor dxDesc,
    Pointer dx) {
    int result = JCudnn.cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    log("cudnnActivationBackward", result, handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    cudnnHandle handle,
    cudnnActivationDescriptor activationDesc,
    Pointer alpha,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    Pointer beta,
    cudnnTensorDescriptor yDesc,
    Pointer y) {
    int result = JCudnn.cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    log("cudnnActivationForward", result, handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnTensorDescriptor aDesc,
    Pointer A,
    Pointer beta,
    cudnnTensorDescriptor cDesc,
    Pointer C) {
    int result = JCudnn.cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
    log("cudnnAddTensor", result, handle, alpha, aDesc, A, beta, cDesc, C);
    handle(result);
    return result;
  }
  
  /**
   * Cudnn reduce tensor int.
   *
   * @param handle               the handle
   * @param reduceTensorDesc     the reduce tensor desc
   * @param indices              the indices
   * @param indicesSizeInBytes   the indices size in bytes
   * @param workspace            the workspace
   * @param workspaceSizeInBytes the workspace size in bytes
   * @param alpha                the alpha
   * @param aDesc                the a desc
   * @param A                    the a
   * @param beta                 the beta
   * @param cDesc                the c desc
   * @param C                    the c
   * @return the int
   */
  public static int cudnnReduceTensor(
    cudnnHandle handle,
    cudnnReduceTensorDescriptor reduceTensorDesc,
    Pointer indices,
    long indicesSizeInBytes,
    Pointer workspace,
    long workspaceSizeInBytes,
    Pointer alpha,
    cudnnTensorDescriptor aDesc,
    Pointer A,
    Pointer beta,
    cudnnTensorDescriptor cDesc,
    Pointer C) {
    int result = JCudnn.cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
    log("cudnnReduceTensor", result, handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
    handle(result);
    return result;
  }
  
  /**
   * Cudnn create reduce tensor descriptor int.
   *
   * @param reduceTensorDesc the reduce tensor desc
   * @return the int
   */
  public static int cudnnCreateReduceTensorDescriptor(
    cudnnReduceTensorDescriptor reduceTensorDesc) {
    int result = JCudnn.cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    log("cudnnCreateReduceTensorDescriptor", result, reduceTensorDesc);
    handle(result);
    return result;
  }
  
  /**
   * Cudnn set reduce tensor descriptor int.
   *
   * @param reduceTensorDesc        the reduce tensor desc
   * @param reduceTensorOp          the reduce tensor op
   * @param reduceTensorCompType    the reduce tensor comp type
   * @param reduceTensorNanOpt      the reduce tensor nan opt
   * @param reduceTensorIndices     the reduce tensor indices
   * @param reduceTensorIndicesType the reduce tensor indices type
   * @return the int
   */
  public static int cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor reduceTensorDesc,
    int reduceTensorOp,
    int reduceTensorCompType,
    int reduceTensorNanOpt,
    int reduceTensorIndices,
    int reduceTensorIndicesType) {
    int result = JCudnn.cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    log("cudnnSetReduceTensorDescriptor", result, reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    handle(result);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnTensorDescriptor dyDesc,
    Pointer dy,
    Pointer beta,
    cudnnTensorDescriptor dbDesc,
    Pointer db) {
    int result = JCudnn.cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
    log("cudnnConvolutionBackwardBias", result, handle, alpha, dyDesc, dy, beta, dbDesc, db);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    Pointer beta,
    cudnnTensorDescriptor yDesc,
    Pointer y) {
    int result = JCudnn.cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
    log("cudnnTransformTensor", result, handle, alpha, xDesc, x, beta, yDesc, y);
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
    cudnnPoolingDescriptor poolingDesc,
    cudnnTensorDescriptor inputTensorDesc,
    int nbDims,
    int[] outputTensorDimA) {
    int result = JCudnn.cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
    log("cudnnGetPoolingNdForwardOutputDim", result, poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
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
    cudnnHandle handle,
    cudnnPoolingDescriptor poolingDesc,
    Pointer alpha,
    cudnnTensorDescriptor yDesc,
    Pointer y,
    cudnnTensorDescriptor dyDesc,
    Pointer dy,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    Pointer beta,
    cudnnTensorDescriptor dxDesc,
    Pointer dx) {
    int result = JCudnn.cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    log("cudnnPoolingBackward", result, handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
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
    cudnnHandle handle,
    cudnnPoolingDescriptor poolingDesc,
    Pointer alpha,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    Pointer beta,
    cudnnTensorDescriptor yDesc,
    Pointer y) {
    int result = JCudnn.cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    log("cudnnPoolingForward", result, handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnFilterDescriptor wDesc,
    Pointer w,
    cudnnTensorDescriptor dyDesc,
    Pointer dy,
    cudnnConvolutionDescriptor convDesc,
    int algo,
    Pointer workSpace,
    long workSpaceSizeInBytes,
    Pointer beta,
    cudnnTensorDescriptor dxDesc,
    Pointer dx) {
    int result = JCudnn.cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    log("cudnnConvolutionBackwardData", result, handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    cudnnTensorDescriptor dyDesc,
    Pointer dy,
    cudnnConvolutionDescriptor convDesc,
    int algo,
    Pointer workSpace,
    long workSpaceSizeInBytes,
    Pointer beta,
    cudnnFilterDescriptor dwDesc,
    Pointer dw) {
    int result = JCudnn.cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    log("cudnnConvolutionBackwardFilter", result, handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
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
    cudnnHandle handle,
    Pointer alpha,
    cudnnTensorDescriptor xDesc,
    Pointer x,
    cudnnFilterDescriptor wDesc,
    Pointer w,
    cudnnConvolutionDescriptor convDesc,
    int algo,
    Pointer workSpace,
    long workSpaceSizeInBytes,
    Pointer beta,
    cudnnTensorDescriptor yDesc,
    Pointer y) {
    int result = JCudnn.cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    log("cudnnConvolutionForward", result, handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    return result;
  }
  
  /**
   * Cuda device reset int.
   *
   * @return the int
   */
  public static int cudaDeviceReset() {
    int result = JCuda.cudaDeviceReset();
    log("cudaDeviceReset", result);
    return result;
  }
  
  /**
   * Cuda free int.
   *
   * @param devPtr the dev ptr
   * @return the int
   */
  public static int cudaFree(Pointer devPtr) {
    int result = JCuda.cudaFree(devPtr);
    log("cudaFree", result, devPtr);
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
  public static int cudaMemcpy(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind) {
    int result = JCuda.cudaMemcpy(dst, src, count, cudaMemcpyKind_kind);
    log("cudaMemcpy", result, dst, src, count, cudaMemcpyKind_kind);
    return result;
  }
  
  /**
   * Cuda malloc int.
   *
   * @param devPtr the dev ptr
   * @param size   the size
   * @return the int
   */
  public static int cudaMalloc(Pointer devPtr, long size) {
    int result = JCuda.cudaMalloc(devPtr, size);
    log("cudaMalloc", result, devPtr, size);
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
  public static int cudaMemset(Pointer mem, int c, long count) {
    int result = JCuda.cudaMemset(mem, c, count);
    log("cudaMemset", result, mem, c, count);
    return result;
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
  public CudaPtr allocateForwardWorkspace(int deviceId, cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
    long sizeInBytesArray[] = {0};
    int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      algorithm, sizeInBytesArray);
    log("cudnnGetConvolutionForwardWorkspaceSize", result, cudnnHandle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      algorithm, sizeInBytesArray);
    handle(result);
    long workspaceSize = sizeInBytesArray[0];
    return alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
  }
  
  /**
   * Init thread.
   */
  public void initThread() {
    setDevice(getDeviceNumber());
    //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    //CuDNN.handle(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
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
  public CudaPtr allocateBackwardFilterWorkspace(int deviceId, cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
    long sizeInBytesArray[] = {0};
    int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
      srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
      algorithm, sizeInBytesArray);
    log("cudnnGetConvolutionBackwardFilterWorkspaceSize", result, cudnnHandle,
      srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
      algorithm, sizeInBytesArray);
    handle(result);
    long workspaceSize = sizeInBytesArray[0];
    return alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
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
  public CudaPtr allocateBackwardDataWorkspace(int deviceId, cudnnTensorDescriptor inputDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor outputDesc, int algorithm) {
    long sizeInBytesArray[] = {0};
    int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
      filterDesc, outputDesc, convDesc, inputDesc,
      algorithm, sizeInBytesArray);
    log("cudnnGetConvolutionBackwardDataWorkspaceSize", result, cudnnHandle,
      filterDesc, outputDesc, convDesc, inputDesc,
      algorithm, sizeInBytesArray);
    handle(result);
    long workspaceSize = sizeInBytesArray[0];
    return alloc(deviceId, 0 < workspaceSize ? workspaceSize : 0);
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
  public int getBackwardFilterAlgorithm(cudnnTensorDescriptor inputDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor outputDesc) {
    int algoArray[] = {-1};
    int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
      inputDesc, outputDesc, convDesc, filterDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    log("cudnnGetConvolutionBackwardFilterAlgorithm", result, cudnnHandle,
      inputDesc, outputDesc, convDesc, filterDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    handle(result);
    return algoArray[0];
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
  public int getBackwardDataAlgorithm(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor weightDesc) {
    int algoArray[] = {-1};
    int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
      filterDesc, srcTensorDesc, convDesc, weightDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    log("cudnnGetConvolutionBackwardDataAlgorithm", result, cudnnHandle,
      filterDesc, srcTensorDesc, convDesc, weightDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    handle(result);
    return algoArray[0];
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
  public int getForwardAlgorithm(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc) {
    int algoArray[] = {-1};
    int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    log("cudnnGetConvolutionForwardAlgorithm", result, cudnnHandle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
    handle(result);
    return algoArray[0];
  }
  
  @Override
  public void finalize() throws Throwable {
    int result = JCudnn.cudnnDestroy(cudnnHandle);
    log("cudnnDestroy", result, cudnnHandle);
    handle(result);
  }
  
  /**
   * Gets device number.
   *
   * @return the device number
   */
  public int getDeviceNumber() {
    return deviceNumber;
  }
  
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceNumber + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
}
