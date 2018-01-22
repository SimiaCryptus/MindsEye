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

import com.simiacryptus.mindseye.lang.PersistanceMode;
import com.simiacryptus.mindseye.lang.RecycleBinLong;
import jcuda.Pointer;
import scala.reflect.internal.util.WeakHashSet;

import java.util.Arrays;
import java.util.function.Supplier;

/**
 * A GPU memory segment
 */
public class ManagedCudaPtr {
  
  /**
   * The constant INSTANCES.
   */
  public static final WeakHashSet<ManagedCudaPtr> INSTANCES = new WeakHashSet<>();
  
  private static final int K = 1024;
  private static final int MiB = K * 1024;
  private static final long GiB = 1024 * MiB;
  private static final long MAX = 1 * GiB;
  
  /**
   * The Size.
   */
  public final long size;
  private final MemoryType type;
  private PersistanceMode persistanceMode;
  private volatile Supplier<CudaPtr> ptrRef;
  private final Precision precision;
  private volatile double[] values;
  
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
   * @param persistanceMode the persistance mode
   */
  protected ManagedCudaPtr(final CudaPtr ptr, PersistanceMode persistanceMode) {this(ptr, persistanceMode, Precision.Double);}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr       the ptr
   * @param precision the precision
   */
  protected ManagedCudaPtr(final CudaPtr ptr, Precision precision) {this(ptr, PersistanceMode.Strong, precision);}
  
  /**
   * Instantiates a new Cuda ptr.
   *
   * @param ptr             the ptr
   * @param persistanceMode the persistance mode
   * @param precision       the precision
   */
  protected ManagedCudaPtr(final CudaPtr ptr, PersistanceMode persistanceMode, Precision precision) {
    super();
    this.size = ptr.size;
    this.type = ptr.getType();
    this.persistanceMode = persistanceMode;
    this.ptrRef = this.persistanceMode.wrap(ptr);
    synchronized (ManagedCudaPtr.INSTANCES) {
      ManagedCudaPtr.INSTANCES.add(this);
    }
    this.precision = precision;
  }
  
  /**
   * Free.
   */
  public void free() {
    Supplier<CudaPtr> ptr = this.ptrRef;
    if (ptr != null) {
      CudaPtr cudaPtr = ptr.get();
      if (null != cudaPtr) cudaPtr.free();
    }
    if (null != values) {
      RecycleBinLong.DOUBLES.recycle(values, values.length);
      values = null;
    }
  }
  
  /**
   * Sets gpu persistance.
   *
   * @param persistanceMode the persistance mode
   * @return the gpu persistance
   */
  public ManagedCudaPtr setGpuPersistance(PersistanceMode persistanceMode) {
    if (persistanceMode != PersistanceMode.Strong && null == getValues()) throw new IllegalStateException();
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
    if (null != values) {
      assert data.length == values.length;
      System.arraycopy(values, 0, data, 0, data.length);
    }
    else {
      getCudaPtr().read(precision, data);
    }
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
    if (null != values) {
      assert data.length == values.length;
      Precision.copy(values, data);
    }
    else {
      getCudaPtr().read(precision, data);
    }
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
    if (null != values) {
      assert data.length == values.length;
      System.arraycopy(data, 0, values, 0, data.length);
    }
    else {
      getCudaPtr().write(precision, data);
    }
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
    if (null != values) {
      assert data.length == values.length;
      Precision.copy(data, values);
    }
    else {
      getCudaPtr().write(precision, data);
    }
    return this;
  }
  
  /**
   * Gets device id.
   *
   * @return the device id
   */
  public Integer getDeviceId() {
    CudaPtr cudaPtr = null == ptrRef ? null : ptrRef.get();
    return null == cudaPtr ? null : cudaPtr.getDeviceId();
  }
  
  
  /**
   * Gets ptr.
   *
   * @return the ptr
   */
  public Pointer getPtr() {
    return getCudaPtr().getPtr();
  }
  
  /**
   * Gets cuda ptr.
   *
   * @return the cuda ptr
   */
  public CudaPtr getCudaPtr() {
    CudaPtr cudaPtr;
    cudaPtr = null == ptrRef ? null : ptrRef.get();
    if (null == cudaPtr) {
      synchronized (this) {
        cudaPtr = null == ptrRef ? null : ptrRef.get();
        if (null == cudaPtr) {
          cudaPtr = CudaPtr.allocate(CuDNN.getDevice(), size, type, true);
          ptrRef = persistanceMode.wrap(cudaPtr);
          assert Arrays.stream(values).allMatch(Double::isFinite);
          cudaPtr.write(precision, values);
        }
      }
    }
    return cudaPtr.assertAlive();
  }
  
  /**
   * Get values double [ ].
   *
   * @return the double [ ]
   */
  public double[] getValues() {
    if (null == values) {
      synchronized (this) {
        if (null == values) {
          CudaPtr cudaPtr = getCudaPtr();
          //CuDNN.setDevice(cudaPtr.getDeviceId());
          values = new double[(int) (size / precision.size)];
          cudaPtr.read(precision, values);
          assert Arrays.stream(values).allMatch(Double::isFinite);
        }
      }
    }
    return values;
  }
  
}
