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

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.util.lang.ResourcePool;
import com.simiacryptus.util.lang.StaticResourcePool;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.*;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.JCudnn.cudnnSetPoolingNdDescriptor;
import static jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS;
import static jcuda.runtime.JCuda.*;

/**
 * The type Cu dnn.
 */
public class CuDNN {
    /**
     * The constant devicePool.
     */
    public static final ResourcePool<CuDNN> devicePool = new ResourcePool<CuDNN>(1) {
        @Override
        public CuDNN create() {
            return new CuDNN();
        }
    };
    /**
     * The constant gpuContexts.
     */
    public static final StaticResourcePool<NNLayer.NNExecutionContext> gpuContexts = new StaticResourcePool<NNLayer.NNExecutionContext>(IntStream.range(0,deviceCount())
            .mapToObj(i->new NNLayer.NNExecutionContext(){
                @Override
                public int getCudaDeviceId() {
                    return i;
                }
            }).collect(Collectors.toList()));

    /**
     * The Cudnn handle.
     */
    public final cudnnHandle cudnnHandle;

    /**
     * Device count int.
     *
     * @return the int
     */
    public static int deviceCount() {
        int[] deviceCount = new int[1];
        handle(cudaGetDeviceCount(deviceCount));
        System.out.println(String.format("Identified %s GPU devices", deviceCount[0]));
        return deviceCount[0];
    }

    /**
     * Instantiates a new Cu dnn.
     */
    protected CuDNN() {
        this.cudnnHandle = new cudnnHandle();
        cudnnCreate(cudnnHandle);
        //cudaSetDevice();
    }

    /**
     * Alloc cu dnn ptr.
     *
     * @param output the output
     * @return the cu dnn ptr
     */
    public static CudaPtr alloc(int deviceId, double[] output) {
        return alloc(deviceId, Sizeof.DOUBLE  * 1l * output.length);
    }

    /**
     * Create pooling descriptor cu dnn resource.
     *
     * @param mode       the mode
     * @param poolDims   the pool dims
     * @param windowSize the window size
     * @param padding    the padding
     * @param stride     the stride
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnPoolingDescriptor> createPoolingDescriptor(int mode, int poolDims, int[] windowSize, int[] padding, int[] stride) {
        cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
        cudnnCreatePoolingDescriptor(poolingDesc);
        cudnnSetPoolingNdDescriptor(poolingDesc,
                mode, CUDNN_PROPAGATE_NAN, poolDims, windowSize,
                padding, stride);
        return new CudaResource<cudnnPoolingDescriptor>(poolingDesc, JCudnn::cudnnDestroyPoolingDescriptor);
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
        handle(cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims));
        return tensorOuputDims;
    }

    /**
     * Handle.
     *
     * @param returnCode the return code
     */
    public static void handle(int returnCode) {
        if(returnCode != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("returnCode = " + cudnnStatus.stringFor(returnCode));
        }
    }

    /**
     * Allocate forward workspace cu dnn ptr.
     *
     * @param srcTensorDesc the src tensor desc
     * @param filterDesc    the filter desc
     * @param convDesc      the conv desc
     * @param dstTensorDesc the dst tensor desc
     * @param algorithm     the algorithm
     * @return the cu dnn ptr
     */
    public CudaPtr allocateForwardWorkspace(int deviceId, cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(deviceId, 0<workspaceSize?workspaceSize:0);
    }

    /**
     * Allocate backward filter workspace cu dnn ptr.
     *
     * @param srcTensorDesc the src tensor desc
     * @param filterDesc    the filter desc
     * @param convDesc      the conv desc
     * @param dstTensorDesc the dst tensor desc
     * @param algorithm     the algorithm
     * @return the cu dnn ptr
     */
    public CudaPtr allocateBackwardFilterWorkspace(int deviceId, cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(deviceId, 0<workspaceSize?workspaceSize:0);
    }

    /**
     * Allocate backward data workspace cu dnn ptr.
     *
     * @param inputDesc  the input desc
     * @param filterDesc the filter desc
     * @param convDesc   the conv desc
     * @param outputDesc the output desc
     * @param algorithm  the algorithm
     * @return the cu dnn ptr
     */
    public CudaPtr allocateBackwardDataWorkspace(int deviceId, cudnnTensorDescriptor inputDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor outputDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                filterDesc, outputDesc, convDesc, inputDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(deviceId, 0<workspaceSize?workspaceSize:0);
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
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                inputDesc, outputDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
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
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                filterDesc, srcTensorDesc, convDesc, weightDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
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
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
        return algoArray[0];
    }

    /**
     * New convolution descriptor cu dnn resource.
     *
     * @param paddingX     the padding x
     * @param paddingY     the padding y
     * @param strideHeight the stride height
     * @param strideWidth  the stride width
     * @param mode         the mode
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnConvolutionDescriptor> newConvolutionDescriptor(int paddingX, int paddingY, int strideHeight, int strideWidth, int mode) {
        cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
        handle(cudnnCreateConvolutionDescriptor(convDesc));
        handle(cudnnSetConvolution2dDescriptor(
            convDesc,
            paddingY, // zero-padding height
            paddingX, // zero-padding width
            strideHeight, // vertical filter stride
            strideWidth, // horizontal filter stride
            1, // upscale the input in x-direction
            1, // upscale the input in y-direction
            mode
        ));
        return new CudaResource<>(convDesc, JCudnn::cudnnDestroyConvolutionDescriptor);
    }

    /**
     * Get stride int [ ].
     *
     * @param array the array
     * @return the int [ ]
     */
    public static int[] getStride(int[] array) {
        return IntStream.range(0, array.length).map(i->IntStream.range(i+1, array.length).map(ii-> array[ii]).reduce((a, b)->a*b).orElse(1)).toArray();
    }

    /**
     * New filter descriptor cu dnn resource.
     *
     * @param dataType       the data type
     * @param tensorLayout   the tensor layout
     * @param outputChannels the output channels
     * @param inputChannels  the input channels
     * @param height         the height
     * @param width          the width
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int outputChannels, int inputChannels, int height, int width) {
        cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
        handle(cudnnCreateFilterDescriptor(filterDesc));
        handle(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width));
        return new CudaResource<>(filterDesc, JCudnn::cudnnDestroyFilterDescriptor);
    }

    /**
     * New filter descriptor cu dnn resource.
     *
     * @param dataType     the data type
     * @param tensorLayout the tensor layout
     * @param dimensions   the dimensions
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int[] dimensions) {
        cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
        handle(cudnnCreateFilterDescriptor(filterDesc));
        handle(cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions));
        return new CudaResource<>(filterDesc, JCudnn::cudnnDestroyFilterDescriptor);
    }

    /**
     * New tensor descriptor cu dnn resource.
     *
     * @param dataType     the data type
     * @param tensorLayout the tensor layout
     * @param batchCount   the batch count
     * @param channels     the channels
     * @param height       the height
     * @param width        the width
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnTensorDescriptor> newTensorDescriptor(int dataType, int tensorLayout, int batchCount, int channels, int height, int width) {
        cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
        handle(cudnnCreateTensorDescriptor(desc));
        handle(cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width));
        return new CudaResource<>(desc, JCudnn::cudnnDestroyTensorDescriptor);
    }

    /**
     * New activation descriptor cu dnn resource.
     *
     * @param mode     the mode
     * @param reluNan  the relu nan
     * @param reluCeil the relu ceil
     * @return the cu dnn resource
     */
    public static CudaResource<cudnnActivationDescriptor> newActivationDescriptor(int mode, int reluNan, double reluCeil) {
        cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
        handle(cudnnCreateActivationDescriptor(desc));
        handle(cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil));
        return new CudaResource<>(desc, JCudnn::cudnnDestroyActivationDescriptor);
    }

    /**
     * Alloc cu dnn ptr.
     *
     * @param size the size
     * @return the cu dnn ptr
     */
    public static CudaPtr alloc(int deviceId, long size) {
        return new CudaPtr(size, deviceId);
    }

    /**
     * Java ptr cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public static CudaPtr javaPtr(int deviceId, double... data) {
        return new CudaPtr(Pointer.to(data), data.length * Sizeof.DOUBLE, deviceId);
    }

    /**
     * Java ptr cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public static CudaPtr javaPtr(int deviceId, float... data) {
        return new CudaPtr(Pointer.to(data), data.length * Sizeof.FLOAT, deviceId);
    }

    /**
     * Write cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public static CudaPtr write(int deviceId, double... data) {
        return new CudaPtr(data.length * Sizeof.DOUBLE, deviceId).write(data);
    }

    /**
     * Write cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public static CudaPtr write(int deviceId, float... data) {
        return new CudaPtr(data.length * Sizeof.FLOAT, deviceId).write(data);
    }

    @Override
    public void finalize() throws Throwable {
        handle(cudnnDestroy(cudnnHandle));
    }
}
