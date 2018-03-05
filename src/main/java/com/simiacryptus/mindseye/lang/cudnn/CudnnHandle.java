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
import com.simiacryptus.mindseye.lang.ReshapedTensorList;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import jcuda.jcudnn.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.function.Consumer;

/**
 * The type Gpu handle.
 */
public class CudnnHandle extends CudaDevice {
  /**
   * The Thread context.
   */
  static final ThreadLocal<CudnnHandle> threadContext = new ThreadLocal<>();
  /**
   * The constant CLEANUP.
   */
  public final LinkedBlockingDeque<CudaResourceBase> cleanupNative = new LinkedBlockingDeque<>();
  /**
   * The Handle.
   */
  @Nullable
  public final jcuda.jcudnn.cudnnHandle handle;
  
  /**
   * Instantiates a new Cu dnn.
   *
   * @param deviceNumber the device number
   */
  CudnnHandle(final int deviceNumber) {
    super(deviceNumber);
    if (0 <= this.deviceId) {
      initThread();
      handle = new cudnnHandle();
      JCudnn.cudnnCreate(handle);
    }
    else {
      handle = null;
    }
    //cudaSetDevice();
  }
  
  /**
   * Gets thread handle.
   *
   * @return the thread handle
   */
  public static CudnnHandle getThreadHandle() {
    return threadContext.get();
  }
  
  /**
   * For each.
   *
   * @param fn the fn
   */
  public static void forEach(@javax.annotation.Nonnull final Consumer<? super CudnnHandle> fn) {
    getPool().getAll().forEach(x -> {
      x.initThread();
      fn.accept(x);
    });
  }
  
  /**
   * Add cuda tensor list.
   *
   * @param left  the left
   * @param right the right
   * @return the cuda tensor list
   */
  @Nonnull
  public CudaTensorList add(final CudaTensorList left, final CudaTensorList right) {
    int length = left.length();
    int[] dimensions = right.getDimensions();
    assert dimensions.length <= 3;
    int d2 = dimensions.length < 3 ? 1 : dimensions[2];
    int d1 = dimensions.length < 2 ? 1 : dimensions[1];
    int d0 = dimensions[0];
    Precision precision = right.getPrecision();
    @Nonnull CudaTensor rPtr = getTensor(right, MemoryType.Device, false);
    @Nonnull CudaTensor lPtr = getTensor(left, MemoryType.Device, false);
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
    @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(left.getPrecision(),
      length, d2, d1, d0,
      d2 * d1 * d0, d1 * d0, d0, 1);
    MemoryType memoryType = MemoryType.Managed;
    @Nonnull final CudaMemory outputPtr = allocate((long) outputDescriptor.nStride * precision.size * length, memoryType, true);
    try {
      CudaMemory lPtrMemory = lPtr.getMemory(this);
      CudaMemory rPtrMemory = rPtr.getMemory(this);
      cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
      lPtrMemory.freeRef();
      rPtrMemory.freeRef();
      return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, dimensions, precision);
    } finally {
      opDescriptor.freeRef();
      rPtr.freeRef();
      lPtr.freeRef();
    }
  }
  
  /**
   * Add in place cuda tensor list.
   *
   * @param left  the left
   * @param right the right
   * @return the cuda tensor list
   */
  public CudaTensorList addInPlace(final CudaTensorList left, final TensorList right) {
    @Nullable final CudaTensor lPtr = left.ptr;//.moveTo(gpu.getDeviceNumber());
    @Nullable final CudaTensor rPtr = getTensor(right, left.getPrecision(), MemoryType.Device, false);//.moveTo(gpu.getDeviceNumber());
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    CudaMemory rPtrMemory = rPtr.getMemory(this);
    CudaMemory lPtrMemory = lPtr.getMemory(this);
    cudnnAddTensor(
      left.getPrecision().getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
      left.getPrecision().getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr());
    lPtrMemory.freeRef();
    rPtrMemory.freeRef();
    rPtr.freeRef();
    return left;
  }
  
  /**
   * Gets cuda ptr.
   *
   * @param data       the data
   * @param precision  the precision
   * @param memoryType the memory type
   * @param dense
   * @return the cuda ptr
   */
  @Nonnull
  public CudaTensor getTensor(@Nonnull final TensorList data, @Nonnull final Precision precision, final MemoryType memoryType, final boolean dense) {
    int[] inputSize = data.getDimensions();
    data.assertAlive();
    if (data instanceof ReshapedTensorList) {
      ReshapedTensorList reshapedTensorList = (ReshapedTensorList) data;
      int[] newDims = reshapedTensorList.getDimensions();
      CudaTensor reshapedTensor = getTensor(reshapedTensorList.getInner(), precision, memoryType, true);
      int channels = newDims.length < 3 ? 1 : newDims[2];
      int height = newDims.length < 2 ? 1 : newDims[1];
      int width = newDims.length < 1 ? 1 : newDims[0];
      CudaTensorDescriptor descriptor = newTensorDescriptor(precision, reshapedTensor.descriptor.batchCount,
        channels, height, width,
        channels * height * width,
        height * width,
        width,
        1
      );
      CudaMemory tensorMemory = reshapedTensor.getMemory(this, memoryType);
      reshapedTensor.freeRef();
      CudaTensor cudaTensor = new CudaTensor(tensorMemory, descriptor, precision);
      tensorMemory.freeRef();
      descriptor.freeRef();
      return cudaTensor;
    }
    if (data instanceof CudaTensorList) {
      if (precision == ((CudaTensorList) data).getPrecision()) {
        @Nonnull CudaTensorList cudaTensorList = (CudaTensorList) data;
        return this.getTensor(cudaTensorList, memoryType, dense);
      }
      else {
        logger.warn("Incompatible precision types in GPU");
      }
    }
    final int listLength = data.length();
    final int elementLength = Tensor.length(data.getDimensions());
    @Nonnull final CudaMemory ptr = this.allocate((long) elementLength * listLength * precision.size, memoryType, true);
    for (int i = 0; i < listLength; i++) {
      Tensor tensor = data.get(i);
      assert null != data;
      assert null != tensor;
      assert Arrays.equals(tensor.getDimensions(), data.getDimensions()) : Arrays.toString(tensor.getDimensions()) + " != " + Arrays.toString(data.getDimensions());
      double[] tensorData = tensor.getData();
      ptr.write(precision, tensorData, (long) i * elementLength);
      tensor.freeRef();
    }
    final int channels = inputSize.length < 3 ? 1 : inputSize[2];
    final int height = inputSize.length < 2 ? 1 : inputSize[1];
    final int width = inputSize.length < 1 ? 1 : inputSize[0];
    @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor descriptor = newTensorDescriptor(precision, data.length(), channels, height, width, channels * height * width, height * width, width, 1);
    return CudaTensor.wrap(ptr, descriptor, precision);
  }
  
  /**
   * The Ptr.
   *
   * @param data       the data
   * @param memoryType the memory type
   * @param dense
   * @return the ptr
   */
  @Nonnull
  public CudaTensor getTensor(@Nonnull final CudaTensorList data, @Nonnull final MemoryType memoryType, final boolean dense) {
    CudaTensor ptr = data.ptr;
    ptr.addRef();
    if ((null == ptr || ptr.isFinalized()) && null != data.heapCopy && !data.heapCopy.isFinalized()) {
      ptr = getTensor(data.heapCopy, data.getPrecision(), memoryType, false);
    }
    if (dense) ptr = ptr.getDenseAndFree(this);
    if (null == ptr) {
      throw new IllegalStateException("No data");
    }
    synchronized (data) {
      if (ptr != data.ptr) {
        if (null != data.ptr) data.ptr.freeRef();
        data.ptr = ptr;
        data.ptr.addRef();
      }
      return ptr;
    }
  }
  
  private TensorList addInPlaceAndFree(final CudaTensorList left, final TensorList right) {
    CudaTensorList cudaTensorList = addInPlace(left, right);
    right.freeRef();
    return cudaTensorList;
  }
  
  /**
   * Add and free tensor list.
   *
   * @param precision the precision
   * @param left      the left
   * @param right     the right
   * @return the tensor list
   */
  @Nonnull
  public TensorList addAndFree(final Precision precision, final TensorList left, final TensorList right) {
    final int[] dimensions = left.getDimensions();
    assert left.length() == right.length();
    assert Tensor.length(left.getDimensions()) == Tensor.length(right.getDimensions());
    int length = left.length();
    assert length == right.length();
    if (left.currentRefCount() == 1 && left instanceof CudaTensorList && ((CudaTensorList) left).ptr.memory.getDeviceId() == getDeviceId())
      return this.addInPlaceAndFree((CudaTensorList) left, right);
    if (right.currentRefCount() == 1 && right instanceof CudaTensorList && ((CudaTensorList) right).ptr.memory.getDeviceId() == getDeviceId())
      return this.addInPlaceAndFree((CudaTensorList) right, left);
    @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
    @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = newTensorDescriptor(precision, length, dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
    @Nullable final CudaTensor lPtr = getTensor(left, precision, MemoryType.Device, false);//.moveTo(gpu.getDeviceNumber());
    @Nullable final CudaTensor rPtr = getTensor(right, precision, MemoryType.Device, false);//.moveTo(gpu.getDeviceNumber());
    assert lPtr.descriptor.batchCount == rPtr.descriptor.batchCount;
    assert lPtr.descriptor.channels == rPtr.descriptor.channels;
    assert lPtr.descriptor.height == rPtr.descriptor.height;
    assert lPtr.descriptor.width == rPtr.descriptor.width;
    @Nonnull final CudaMemory outputPtr = allocate(outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
    CudaMemory lPtrMemory = lPtr.getMemory(this);
    CudaMemory rPtrMemory = rPtr.getMemory(this);
    cudnnOpTensor(opDescriptor.getPtr(),
      precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
      precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
      precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
    lPtrMemory.freeRef();
    rPtrMemory.freeRef();
    Arrays.stream(new ReferenceCounting[]{lPtr, rPtr, opDescriptor, left, right}).forEach(ReferenceCounting::freeRef);
    return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, dimensions, precision);
  }
  
  
  /**
   * Cudnn activation forward int.
   *
   * @param activationDesc the activation desc
   * @param alpha          the alpha
   * @param xDesc          the x desc
   * @param x              the x
   * @param beta           the beta
   * @param yDesc          the y desc
   * @param y              the y
   * @return the int
   */
  public int cudnnActivationForward(
    final cudnnActivationDescriptor activationDesc,
    final CudaPointer alpha,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final CudaPointer beta,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnActivationForward(this.handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnActivationForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnActivationForward", result, new Object[]{this, activationDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }
  
  /**
   * Cudnn add tensor int.
   *
   * @param alpha the alpha
   * @param aDesc the a desc
   * @param A     the a
   * @param beta  the beta
   * @param cDesc the c desc
   * @param C     the c
   * @return the int
   */
  public int cudnnAddTensor(
    final CudaPointer alpha,
    final cudnnTensorDescriptor aDesc,
    final CudaPointer A,
    final CudaPointer beta,
    final cudnnTensorDescriptor cDesc,
    final CudaPointer C) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnAddTensor(this.handle, alpha, aDesc, A, beta, cDesc, C);
    cudnnAddTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnAddTensor", result, new Object[]{this, alpha, aDesc, A, beta, cDesc, C});
    CudaSystem.handle(result);
    return result;
  }
  
  /**
   * Cudnn convolution backward bias int.
   *
   * @param alpha  the alpha
   * @param dyDesc the dy desc
   * @param dy     the dy
   * @param beta   the beta
   * @param dbDesc the db desc
   * @param db     the db
   * @return the int
   */
  public int cudnnConvolutionBackwardBias(
    final CudaPointer alpha,
    final cudnnTensorDescriptor dyDesc,
    final CudaPointer dy,
    final CudaPointer beta,
    final cudnnTensorDescriptor dbDesc,
    final CudaPointer db) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardBias(this.handle, alpha, dyDesc, dy, beta, dbDesc, db);
    cudnnConvolutionBackwardBias_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnConvolutionBackwardBias", result, new Object[]{this, alpha, dyDesc, dy, beta, dbDesc, db});
    return result;
  }
  
  /**
   * Cudnn convolution backward data int.
   *
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
  public int cudnnConvolutionBackwardData(
    final CudaPointer alpha,
    final cudnnFilterDescriptor wDesc,
    final CudaPointer w,
    final cudnnTensorDescriptor dyDesc,
    final CudaPointer dy,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final CudaPointer workSpace,
    final long workSpaceSizeInBytes,
    final CudaPointer beta,
    final cudnnTensorDescriptor dxDesc,
    final CudaPointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardData(this.handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    cudnnConvolutionBackwardData_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnConvolutionBackwardData", result, new Object[]{this, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx});
    return result;
  }
  
  /**
   * Cudnn convolution backward filter int.
   *
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
  public int cudnnConvolutionBackwardFilter(
    final CudaPointer alpha,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final cudnnTensorDescriptor dyDesc,
    final CudaPointer dy,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final CudaPointer workSpace,
    final long workSpaceSizeInBytes,
    final CudaPointer beta,
    final cudnnFilterDescriptor dwDesc,
    final CudaPointer dw) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionBackwardFilter(this.handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    cudnnConvolutionBackwardFilter_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnConvolutionBackwardFilter", result, new Object[]{this, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw});
    return result;
  }
  
  /**
   * Cudnn convolution forward int.
   *
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
  public int cudnnConvolutionForward(
    final CudaPointer alpha,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final cudnnFilterDescriptor wDesc,
    final CudaPointer w,
    final cudnnConvolutionDescriptor convDesc,
    final int algo,
    final CudaPointer workSpace,
    final long workSpaceSizeInBytes,
    final CudaPointer beta,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnConvolutionForward(this.handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    cudnnConvolutionForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnConvolutionForward", result, new Object[]{this, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y});
    return result;
  }
  
  /**
   * Cudnn op tensor int.
   *
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
  public int cudnnOpTensor(
    final cudnnOpTensorDescriptor opTensorDesc,
    final CudaPointer alpha1,
    final cudnnTensorDescriptor aDesc,
    final CudaPointer A,
    final CudaPointer alpha2,
    final cudnnTensorDescriptor bDesc,
    final CudaPointer B,
    final CudaPointer beta,
    final cudnnTensorDescriptor cDesc,
    final CudaPointer C) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnOpTensor(this.handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    cudnnOpTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnOpTensor", result, new Object[]{this, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C});
    return result;
  }
  
  /**
   * Cudnn pooling backward int.
   *
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
  public int cudnnPoolingBackward(
    final cudnnPoolingDescriptor poolingDesc,
    final CudaPointer alpha,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y,
    final cudnnTensorDescriptor dyDesc,
    final CudaPointer dy,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final CudaPointer beta,
    final cudnnTensorDescriptor dxDesc,
    final CudaPointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnPoolingBackward(this.handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    cudnnPoolingBackward_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnPoolingBackward", result, new Object[]{this, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }
  
  /**
   * Cudnn pooling forward int.
   *
   * @param poolingDesc the pooling desc
   * @param alpha       the alpha
   * @param xDesc       the x desc
   * @param x           the x
   * @param beta        the beta
   * @param yDesc       the y desc
   * @param y           the y
   * @return the int
   */
  public int cudnnPoolingForward(
    final cudnnPoolingDescriptor poolingDesc,
    final CudaPointer alpha,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final CudaPointer beta,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnPoolingForward(this.handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    cudnnPoolingForward_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnPoolingForward", result, new Object[]{this, poolingDesc, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }
  
  /**
   * Cudnn transform tensor int.
   *
   * @param alpha the alpha
   * @param xDesc the x desc
   * @param x     the x
   * @param beta  the beta
   * @param yDesc the y desc
   * @param y     the y
   * @return the int
   */
  public int cudnnTransformTensor(
    final CudaPointer alpha,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final CudaPointer beta,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnTransformTensor(this.handle, alpha, xDesc, x, beta, yDesc, y);
    cudnnTransformTensor_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnTransformTensor", result, new Object[]{this, alpha, xDesc, x, beta, yDesc, y});
    return result;
  }
  
  /**
   * Allocate backward data workspace cuda ptr.
   *
   * @param inputDesc  the input desc
   * @param filterDesc the filter desc
   * @param convDesc   the conv desc
   * @param outputDesc the output desc
   * @param algorithm  the algorithm
   * @return the cuda ptr
   */
  public CudaMemory allocateBackwardDataWorkspace(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc, final int algorithm) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
      filterDesc, outputDesc, convDesc, inputDesc,
      algorithm, sizeInBytesArray);
    allocateBackwardDataWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardDataWorkspaceSize", result, new Object[]{this, filterDesc, outputDesc, convDesc, inputDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return this.allocate(Math.max(1, size), MemoryType.Device, true);
  }
  
  /**
   * Allocate backward filter workspace cuda ptr.
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public CudaMemory allocateBackwardFilterWorkspace(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
      srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
      algorithm, sizeInBytesArray);
    allocateBackwardFilterWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardFilterWorkspaceSize", result, new Object[]{this, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return allocate(Math.max(1, size), MemoryType.Device, true);
  }
  
  /**
   * Allocate forward workspace cuda ptr.
   *
   * @param srcTensorDesc the src tensor desc
   * @param filterDesc    the filter desc
   * @param convDesc      the conv desc
   * @param dstTensorDesc the dst tensor desc
   * @param algorithm     the algorithm
   * @return the cuda ptr
   */
  public CudaMemory allocateForwardWorkspace(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final int algorithm) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final long sizeInBytesArray[] = {0};
    final int result = JCudnn.cudnnGetConvolutionForwardWorkspaceSize(handle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      algorithm, sizeInBytesArray);
    allocateForwardWorkspace_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionForwardWorkspaceSize", result, new Object[]{this, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algorithm, sizeInBytesArray});
    CudaSystem.handle(result);
    final long size = sizeInBytesArray[0];
    return this.allocate(Math.max(1, size), MemoryType.Device, true);
  }
  
  /**
   * Gets backward data algorithm.
   *
   * @param inputDesc          the src tensor desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param outputDesc         the weight desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the backward data algorithm
   */
  public int getBackwardDataAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc, final long memoryLimitInBytes) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(handle,
      filterDesc, inputDesc, convDesc, outputDesc,
      cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardDataAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardDataAlgorithm", result, new Object[]{this, filterDesc, inputDesc, convDesc, outputDesc, cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets backward filter algorithm.
   *
   * @param inputDesc          the input desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param outputDesc         the output desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the backward filter algorithm
   */
  public int getBackwardFilterAlgorithm(final cudnnTensorDescriptor inputDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor outputDesc, final long memoryLimitInBytes) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(handle,
      inputDesc, outputDesc, convDesc, filterDesc,
      cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getBackwardFilterAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionBackwardFilterAlgorithm", result, new Object[]{this, inputDesc, outputDesc, convDesc, filterDesc, cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }
  
  /**
   * Gets forward algorithm.
   *
   * @param srcTensorDesc      the src tensor desc
   * @param filterDesc         the filter desc
   * @param convDesc           the conv desc
   * @param dstTensorDesc      the dst tensor desc
   * @param memoryLimitInBytes the memory limit in bytes
   * @return the forward algorithm
   */
  public int getForwardAlgorithm(final cudnnTensorDescriptor srcTensorDesc, final cudnnFilterDescriptor filterDesc, final cudnnConvolutionDescriptor convDesc, final cudnnTensorDescriptor dstTensorDesc, final long memoryLimitInBytes) {
    long startTime = System.nanoTime();
    @javax.annotation.Nonnull final int algoArray[] = {-1};
    final int result = JCudnn.cudnnGetConvolutionForwardAlgorithm(handle,
      srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
      cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray);
    getForwardAlgorithm_execution.accept((System.nanoTime() - startTime) / 1e9);
    log("cudnnGetConvolutionForwardAlgorithm", result, new Object[]{this, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, memoryLimitInBytes, algoArray});
    CudaSystem.handle(result);
    return algoArray[0];
  }
  
  /**
   * Cudnn activation backward int.
   *
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
  public int cudnnActivationBackward(
    final cudnnActivationDescriptor activationDesc,
    final CudaPointer alpha,
    final cudnnTensorDescriptor yDesc,
    final CudaPointer y,
    final cudnnTensorDescriptor dyDesc,
    final CudaPointer dy,
    final cudnnTensorDescriptor xDesc,
    final CudaPointer x,
    final CudaPointer beta,
    final cudnnTensorDescriptor dxDesc,
    final CudaPointer dx) {
    long startTime = System.nanoTime();
    final int result = JCudnn.cudnnActivationBackward(this.handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    cudnnActivationBackward_execution.accept((System.nanoTime() - startTime) / 1e9);
    this.dirty();
    log("cudnnActivationBackward", result, new Object[]{this, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx});
    return result;
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" + deviceId + "; " + deviceName + "}@" + Long.toHexString(System.identityHashCode(this));
  }
  
  @Override
  public void finalize() throws Throwable {
    final int result = JCudnn.cudnnDestroy(handle);
    log("cudnnDestroy", result, new Object[]{handle});
    CudaSystem.handle(result);
  }
  
  @Override
  protected void cleanup() {
    super.cleanup();
    ArrayList<CudaResourceBase> objsToFree = new ArrayList<>();
    cleanupNative.drainTo(objsToFree);
    objsToFree.stream().forEach(CudaResourceBase::release);
  }
}
