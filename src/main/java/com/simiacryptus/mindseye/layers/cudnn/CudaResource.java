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

import java.util.function.ToIntFunction;

/**
 * A managed resource containing a native CuDNN resource, bound to its lifecycle with a prearranged destructor.
 *
 * @param <T> the type parameter
 */
public class CudaResource<T> extends CudaResourceBase<T> {
  
  private final ToIntFunction<T> destructor;
  
  /**
   * Instantiates a new Cuda resource.
   *
   * @param obj        the obj
   * @param destructor the destructor
   */
  protected CudaResource(final T obj, final ToIntFunction<T> destructor) {
    super(obj);
    this.destructor = destructor;
  }
  
  /**
   * Free.
   */
  protected void free() {
    try {
      if (isActiveObj()) {
        CuDNN.handle(this.destructor.applyAsInt(ptr));
      }
    } catch (final Throwable e) {
      //new ComponentException("Error freeing resource " + this, e).printStackTrace(System.err);
    }
  }
  
}
