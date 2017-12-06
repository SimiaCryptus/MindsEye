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

import com.simiacryptus.mindseye.lang.ComponentException;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.ToIntFunction;

/**
 * The type Cuda resource.
 *
 * @param <T> the type parameter
 */
public class CudaResource<T> {
  
  /**
   * The constant gpuGeneration.
   */
  public static AtomicInteger gpuGeneration = new AtomicInteger(0);
  /**
   * The constant debugLifecycle.
   */
  public static boolean debugLifecycle = true;
  /**
   * The Obj generation.
   */
  public final int objGeneration = gpuGeneration.get();
  /**
   * The Created by.
   */
  public final StackTraceElement[] createdBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
  private final T ptr;
  private final ToIntFunction<T> destructor;
  //private final StackTraceElement[] createdBy = Thread.currentThread().getStackTrace();
  private final int device = CuDNN.getDevice();
  /**
   * The Finalized by.
   */
  public StackTraceElement[] finalizedBy = null;
  private volatile boolean finalized = false;
  
  /**
   * Instantiates a new Cuda resource.
   *
   * @param obj        the obj
   * @param destructor the destructor
   */
  protected CudaResource(T obj, ToIntFunction<T> destructor) {
    this.ptr = obj;
    this.destructor = destructor;
  }
  
  /**
   * Is finalized boolean.
   *
   * @return the boolean
   */
  public boolean isFinalized() {
    return finalized;
  }
  
  @Override
  public synchronized void finalize() {
    try {
      if (!this.finalized && isActiveObj()) {
        if (null != this.destructor) free();
        finalizedBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
        this.finalized = true;
      }
      super.finalize();
    } catch (Throwable e) {
      new ComponentException("Error freeing resource " + this, e).printStackTrace(System.err);
    }
  }
  
  /**
   * Is active obj boolean.
   *
   * @return the boolean
   */
  public boolean isActiveObj() {
    return objGeneration == gpuGeneration.get();
  }
  
  /**
   * Free.
   */
  protected void free() {
    try {
      if (isActiveObj()) {
        CuDNN.handle(this.destructor.applyAsInt(ptr));
      }
    } catch (Throwable e) {
      //new ComponentException("Error freeing resource " + this, e).printStackTrace(System.err);
    }
  }
  
  /**
   * Gets ptr.
   *
   * @return the ptr
   */
  public T getPtr() {
    if (isFinalized()) return null;
    return ptr;
  }
}
