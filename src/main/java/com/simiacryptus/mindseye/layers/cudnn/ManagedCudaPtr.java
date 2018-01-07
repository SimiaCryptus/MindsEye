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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.PersistanceMode;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr.MemoryType;
import jcuda.Pointer;
import scala.reflect.internal.util.WeakHashSet;

import java.util.function.Supplier;

/**
 * A GPU memory segment
 */
public class ManagedCudaPtr {
  
  public static final WeakHashSet<ManagedCudaPtr> INSTANCES = new WeakHashSet<>();
  
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  private static final long MAX = 1 * GiB;
  
  /**
   * The Size.
   */
  public final long size;
  private final int deviceId;
  private final MemoryType type;
  private final boolean dirty;
  private PersistanceMode persistanceMode;
  private volatile Supplier<CudaPtr> ptrRef;
  private volatile byte[] bytes;
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr the ptr
   */
  protected ManagedCudaPtr(final CudaPtr ptr) {this(ptr, PersistanceMode.Strong);}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr             the ptr
   * @param persistanceMode
   */
  protected ManagedCudaPtr(final CudaPtr ptr, PersistanceMode persistanceMode) {
    super();
    this.size = ptr.size;
    this.deviceId = ptr.getDeviceId();
    this.type = ptr.getType();
    this.persistanceMode = persistanceMode;
    this.ptrRef = this.persistanceMode.wrap(ptr);
    this.dirty = false;
    ManagedCudaPtr.INSTANCES.add(this);
  }
  
  protected void free() {
    Supplier<CudaPtr> ptr = this.ptrRef;
    if (ptr != null) {
      CudaPtr cudaPtr = ptr.get();
      if (null != cudaPtr) cudaPtr.free();
    }
    bytes = null;
  }
  
  public ManagedCudaPtr setGpuPersistance(PersistanceMode persistanceMode) {
    if (persistanceMode != PersistanceMode.Strong && null == getBytes()) throw new IllegalStateException();
    if (persistanceMode == PersistanceMode.Strong && null == getPtr()) throw new IllegalStateException();
    if (ptrRef != null) {
      CudaPtr cudaPtr = ptrRef.get();
      if (null != cudaPtr) {
        ptrRef = persistanceMode.wrap(cudaPtr);
      }
    }
    this.persistanceMode = persistanceMode;
    return this;
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public ManagedCudaPtr read(final Precision precision, final double[] data) {
    getCudaPtr().read(precision, data);
    return this;
  }
  
  /**
   * Read cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public ManagedCudaPtr read(final Precision precision, final float[] data) {
    getCudaPtr().read(precision, data);
    return this;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public ManagedCudaPtr write(final Precision precision, final double[] data) {
    getCudaPtr().write(precision, data);
    return this;
  }
  
  /**
   * Write cuda ptr.
   *
   * @param precision the precision
   * @param data      the data
   * @return the cuda ptr
   */
  public ManagedCudaPtr write(final Precision precision, final float[] data) {
    getCudaPtr().write(precision, data);
    return this;
  }
  
  /**
   * Gets device id.
   *
   * @return the device id
   */
  public int getDeviceId() {
    return deviceId;
  }
  
  
  public Pointer getPtr() {
    return getCudaPtr().getPtr();
  }
  
  public CudaPtr getCudaPtr() {
    CudaPtr cudaPtr;
    cudaPtr = null == ptrRef ? null : ptrRef.get();
    if (null == cudaPtr) {
      synchronized (this) {
        cudaPtr = null == ptrRef ? null : ptrRef.get();
        if (null == cudaPtr) {
          cudaPtr = new CudaPtr(size, deviceId, type, dirty);
          ptrRef = persistanceMode.wrap(cudaPtr);
        }
      }
    }
    return cudaPtr;
  }
  
  public byte[] getBytes() {
    if (null == bytes) {
      synchronized (this) {
        if (null == bytes) {
          double[] bytes = new double[(int) (size / Precision.Double.size)];
          getCudaPtr().read(Precision.Double, bytes);
        }
      }
    }
    return bytes;
  }
  
}
