package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.util.lang.ResourcePool;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;

import java.util.function.Consumer;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.JCudnn.cudnnSetPoolingNdDescriptor;
import static jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemset;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

public class CuDNN {
    public static final ResourcePool<CuDNN> devicePool = new ResourcePool<CuDNN>(1) {
        @Override
        public CuDNN create() {
            return new CuDNN();
        }
    };
    public final cudnnHandle cudnnHandle;

    protected CuDNN() {
        this.cudnnHandle = new cudnnHandle();
        cudnnCreate(cudnnHandle);
    }

    public static CuDNNPtr alloc(double[] output) {
        return alloc(Sizeof.DOUBLE * output.length);
    }

    public static CuDNNResource<cudnnPoolingDescriptor> createPoolingDescriptor(int mode, int poolDims, int[] windowSize, int[] padding, int[] stride) {
        cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
        cudnnCreatePoolingDescriptor(poolingDesc);
        cudnnSetPoolingNdDescriptor(poolingDesc,
                mode, CUDNN_PROPAGATE_NAN, poolDims, windowSize,
                padding, stride);
        return new CuDNNResource<cudnnPoolingDescriptor>(poolingDesc, JCudnn::cudnnDestroyPoolingDescriptor);
    }

    public static class CuDNNResource<T> {

        private final T ptr;
        private final Consumer<T> destructor;
        private boolean finalized = false;

        protected CuDNNResource(T obj, Consumer<T> destructor) {
            this.ptr = obj;
            this.destructor = destructor;
        }

        public boolean isFinalized() {
            return finalized;
        }

        @Override
        protected synchronized void finalize() {
            if(!this.finalized) {
                if(null != this.destructor) this.destructor.accept(ptr);
                this.finalized = true;
            }
            try {
                super.finalize();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }

        public T getPtr() {
            if(isFinalized()) return null;
            return ptr;
        }
    }

    public static class CuDNNPtr extends CuDNNResource<Pointer> {
        public final long size;

        protected CuDNNPtr(long size) {
            super(new Pointer(), JCuda::cudaFree);
            this.size = size;
            try {
                handle(cudaMalloc(this.getPtr(), size));
                handle(cudaMemset(this.getPtr(), 0, size));

            } catch (Exception e) {
                throw new RuntimeException("Error allocating " + size + " bytes", e);
            }
        }

        protected CuDNNPtr(Pointer ptr, long size) {
            super(ptr, x->{});
            this.size = size;
        }

        public CuDNNPtr write(float[] data) {
            if(this.size != data.length * Sizeof.FLOAT) throw new IllegalArgumentException();
            handle(cudaMemcpy(getPtr(), Pointer.to(data), size, cudaMemcpyHostToDevice));
            return this;
        }

        public CuDNNPtr write(double[] data) {
            if(this.size != data.length * Sizeof.DOUBLE) throw new IllegalArgumentException();
            handle(cudaMemcpy(getPtr(), Pointer.to(data), size, cudaMemcpyHostToDevice));
            return this;
        }

        public CuDNNPtr read(double[] data) {
            if(this.size != data.length * Sizeof.DOUBLE) throw new IllegalArgumentException(this.size +" != " + data.length * Sizeof.DOUBLE);
            handle(cudaMemcpy(Pointer.to(data), getPtr(), size, cudaMemcpyDeviceToHost));
            return this;
        }

        public CuDNNPtr read(float[] data) {
            if(this.size != data.length * Sizeof.FLOAT) throw new IllegalArgumentException();
            handle(cudaMemcpy(Pointer.to(data), getPtr(), size, cudaMemcpyDeviceToHost));
            return this;
        }
    }

    protected static int[] getOutputDims(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc) {
        int[] tensorOuputDims = new int[4];
        handle(cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorOuputDims.length, tensorOuputDims));
        return tensorOuputDims;
    }

    public static void handle(int returnCode) {
        if(returnCode != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("returnCode = " + cudnnStatus.stringFor(returnCode));
        }
    }

    protected CuDNNPtr allocateForwardWorkspace(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(0<workspaceSize?workspaceSize:0);
    }

    protected CuDNNPtr allocateBackwardFilterWorkspace(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(0<workspaceSize?workspaceSize:0);
    }

    protected CuDNNPtr allocateBackwardDataWorkspace(cudnnTensorDescriptor inputDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor outputDesc, int algorithm) {
        long sizeInBytesArray[] = { 0 };
        handle(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                filterDesc, outputDesc, convDesc, inputDesc,
                algorithm, sizeInBytesArray));
        long workspaceSize = sizeInBytesArray[0];
        return alloc(0<workspaceSize?workspaceSize:0);
    }

    protected int getBackwardFilterAlgorithm(cudnnTensorDescriptor inputDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor outputDesc) {
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                inputDesc, outputDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
        return algoArray[0];
    }

    protected int getBackwardDataAlgorithm(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor weightDesc) {
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                filterDesc, srcTensorDesc, convDesc, weightDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
        return algoArray[0];
    }

    protected int getForwardAlgorithm(cudnnTensorDescriptor srcTensorDesc, cudnnFilterDescriptor filterDesc, cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor dstTensorDesc) {
        int algoArray[] = { -1 };
        handle(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray));
        return algoArray[0];
    }

    protected static CuDNNResource<cudnnConvolutionDescriptor> newConvolutionDescriptor(int paddingX,int paddingY, int strideHeight, int strideWidth, int mode) {
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
        return new CuDNNResource<>(convDesc, JCudnn::cudnnDestroyConvolutionDescriptor);
    }

    private static int[] getStride(int[] array) {
        return IntStream.range(0, array.length).map(i->IntStream.range(i+1, array.length).map(ii-> array[ii]).reduce((a, b)->a*b).orElse(1)).toArray();
    }

    public static CuDNNResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int outputChannels, int inputChannels, int height, int width) {
        cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
        handle(cudnnCreateFilterDescriptor(filterDesc));
        handle(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorLayout, outputChannels, inputChannels, height, width));
        return new CuDNNResource<>(filterDesc, JCudnn::cudnnDestroyFilterDescriptor);
    }

    public static CuDNNResource<cudnnFilterDescriptor> newFilterDescriptor(int dataType, int tensorLayout, int[] dimensions) {
        cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
        handle(cudnnCreateFilterDescriptor(filterDesc));
        handle(cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorLayout, dimensions.length, dimensions));
        return new CuDNNResource<>(filterDesc, JCudnn::cudnnDestroyFilterDescriptor);
    }
    public static CuDNNResource<cudnnTensorDescriptor> newTensorDescriptor(int dataType, int tensorLayout, int batchCount, int channels, int height, int width) {
        cudnnTensorDescriptor desc = new cudnnTensorDescriptor();
        handle(cudnnCreateTensorDescriptor(desc));
        handle(cudnnSetTensor4dDescriptor(desc, tensorLayout, dataType, batchCount, channels, height, width));
        return new CuDNNResource<>(desc, JCudnn::cudnnDestroyTensorDescriptor);
    }

    public static CuDNNResource<cudnnActivationDescriptor> newActivationDescriptor(int mode, int reluNan, double reluCeil) {
        cudnnActivationDescriptor desc = new cudnnActivationDescriptor();
        handle(cudnnCreateActivationDescriptor(desc));
        handle(cudnnSetActivationDescriptor(desc, mode, reluNan, reluCeil));
        return new CuDNNResource<>(desc, JCudnn::cudnnDestroyActivationDescriptor);
    }

    public static CuDNNPtr alloc(long size) {
        return new CuDNNPtr(size);
    }

    public static CuDNNPtr javaPtr(double... data) {
        return new CuDNNPtr(Pointer.to(data), data.length * Sizeof.DOUBLE);
    }

    public static CuDNNPtr javaPtr(float... data) {
        return new CuDNNPtr(Pointer.to(data), data.length * Sizeof.FLOAT);
    }

    public static CuDNNPtr write(double... data) {
        return new CuDNNPtr(data.length * Sizeof.DOUBLE).write(data);
    }

    public static CuDNNPtr write(float... data) {
        return new CuDNNPtr(data.length * Sizeof.FLOAT).write(data);
    }

    @Override
    protected void finalize() throws Throwable {
        handle(cudnnDestroy(cudnnHandle));
    }
}
