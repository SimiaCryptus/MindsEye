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

import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Cuda resource base.
 *
 * @param <T> the type parameter
 */
public abstract class CudaResourceBase<T> extends ReferenceCountingBase {
  private static final Logger logger = LoggerFactory.getLogger(CudaResourceBase.class);
  /**
   * The constant debugLifecycle.
   */
  public static boolean debugLifecycle = true;
  /**
   * The constant gpuGeneration.
   */
  public static AtomicInteger gpuGeneration = new AtomicInteger(0);
  /**
   * The Obj generation.
   */
  public final int objGeneration = CudaResourceBase.gpuGeneration.get();
  /**
   * The Ptr.
   */
  protected final T ptr;
  
  /**
   * Instantiates a new Cuda resource base.
   *
   * @param obj the obj
   */
  public CudaResourceBase(final T obj) {this.ptr = obj;}
  
  /**
   * Gets ptr.
   *
   * @return the ptr
   */
  public T getPtr() {
    assertAlive();
    return ptr;
  }
  
  /**
   * Free.
   */
  protected abstract void _free();
  
  /**
   * Is active obj boolean.
   *
   * @return the boolean
   */
  public boolean isActiveObj() {
    return objGeneration == CudaResourceBase.gpuGeneration.get();
  }
  
}
