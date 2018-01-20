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

import static jcuda.runtime.cudaLimit.*;

/**
 * The enum Device limits.
 */
public enum DeviceLimits {
  /**
   * Stack size device limits.
   */
  StackSize(cudaLimitStackSize),
  /**
   * Fifo size device limits.
   */
  FifoSize(cudaLimitPrintfFifoSize),
  /**
   * Heap size device limits.
   */
  HeapSize(cudaLimitMallocHeapSize),
  /**
   * Sync depth device limits.
   */
  SyncDepth(cudaLimitDevRuntimeSyncDepth),
  /**
   * Pending launch device limits.
   */
  PendingLaunch(cudaLimitDevRuntimePendingLaunchCount);
  
  /**
   * The Id.
   */
  public final int id;
  
  DeviceLimits(int id) {
    this.id = id;
  }
  
  /**
   * Get long.
   *
   * @return the long
   */
  public long get() {
    return CuDNN.cudaDeviceGetLimit(id);
  }
  
  /**
   * Set.
   *
   * @param value the value
   */
  public void set(long value) {
    CuDNN.cudaDeviceSetLimit(id, value);
  }
}
