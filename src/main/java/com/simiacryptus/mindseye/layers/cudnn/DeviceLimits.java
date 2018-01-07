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

import static jcuda.runtime.cudaLimit.*;

public enum DeviceLimits {
  StackSize(cudaLimitStackSize),
  FifoSize(cudaLimitPrintfFifoSize),
  HeapSize(cudaLimitMallocHeapSize),
  SyncDepth(cudaLimitDevRuntimeSyncDepth),
  PendingLaunch(cudaLimitDevRuntimePendingLaunchCount);
  
  public final int id;
  
  DeviceLimits(int id) {
    this.id = id;
  }
  
  public long get() {
    return CuDNN.cudaDeviceGetLimit(id);
  }
  
  public void set(long value) {
    CuDNN.handle(CuDNN.cudaDeviceSetLimit(id, value));
  }
}
