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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

import java.util.HashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

/**
 * The type Cu dnn ptr.
 */
public class CudaPtr extends CudaResource<Pointer> {
    
    public static final LoadingCache<Integer, AtomicLong> USED_MEMORY = CacheBuilder.newBuilder().build(new CacheLoader<Integer,AtomicLong>() {
        @Override
        public AtomicLong load(Integer integer) throws Exception {
            return new AtomicLong(0);
        }
    });
    private static final long MAX = 4 * 1024 * 1024 * 1024;
    
    /**
     * The Size.
     */
    public final long size;
    private final int deviceId;

    /**
     * Instantiates a new Cu dnn ptr.
     *
     * @param size the size
     */
    protected CudaPtr(long size, int deviceId) {
        super(new Pointer(), JCuda::cudaFree);
        this.size = size;
        this.deviceId = deviceId;
        AtomicLong devivceMemCtr;
        try {
            devivceMemCtr = USED_MEMORY.get(deviceId);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
        
        if(devivceMemCtr.get() + size > MAX) {
            System.gc();
            if(devivceMemCtr.get() + size > MAX) {
                System.err.println(String.format("High device memory usage: %s on device %s", devivceMemCtr.get(), deviceId) );
            }
        }
        try {
            CuDNN.handle(cudaMalloc(this.getPtr(), size));
        } catch (Exception e) {
            try {
                System.gc(); // Force any dead objects to be finalized
                CuDNN.handle(cudaMalloc(this.getPtr(), size));
            } catch (Exception e2) {
                throw new RuntimeException(String.format("Error allocating %s bytes; %s currently allocated to device %s", size, devivceMemCtr.get(), deviceId), e2);
            }
        }
        devivceMemCtr.addAndGet(size);
        CuDNN.handle(cudaMemset(this.getPtr(), 0, size));
    }
    
    @Override
    protected void free() {
        super.free();
        try {
            USED_MEMORY.get(deviceId).addAndGet(-size);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
    
    /**
     * Instantiates a new Cu dnn ptr.
     *
     * @param ptr  the ptr
     * @param size the size
     */
    protected CudaPtr(Pointer ptr, long size, int deviceId) {
        super(ptr, x->0);
        this.size = size;
        this.deviceId = deviceId;
    }

    /**
     * Write cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public CudaPtr write(float[] data) {
        if(this.size != data.length * Sizeof.FLOAT) throw new IllegalArgumentException();
        CuDNN.handle(cudaMemcpy(getPtr(), Pointer.to(data), size, cudaMemcpyHostToDevice));
        return this;
    }

    /**
     * Write cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public CudaPtr write(double[] data) {
        if(this.size != data.length * Sizeof.DOUBLE) throw new IllegalArgumentException();
        CuDNN.handle(cudaMemcpy(getPtr(), Pointer.to(data), size, cudaMemcpyHostToDevice));
        return this;
    }

    /**
     * Read cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public CudaPtr read(double[] data) {
        if(this.size != data.length * Sizeof.DOUBLE) throw new IllegalArgumentException(this.size +" != " + data.length * Sizeof.DOUBLE);
        CuDNN.handle(cudaMemcpy(Pointer.to(data), getPtr(), size, cudaMemcpyDeviceToHost));
        return this;
    }

    /**
     * Read cu dnn ptr.
     *
     * @param data the data
     * @return the cu dnn ptr
     */
    public CudaPtr read(float[] data) {
        if(this.size != data.length * Sizeof.FLOAT) throw new IllegalArgumentException();
        CuDNN.handle(cudaMemcpy(Pointer.to(data), getPtr(), size, cudaMemcpyDeviceToHost));
        return this;
    }
}
