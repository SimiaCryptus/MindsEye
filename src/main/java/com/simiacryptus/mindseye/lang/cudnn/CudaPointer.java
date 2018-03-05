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

import jcuda.Pointer;

public class CudaPointer extends Pointer {
  
  public final int deviceId = CudaSystem.getThreadDeviceId();
  
  public CudaPointer(final Pointer other) {
    super(other);
  }
  
  public CudaPointer(final Pointer other, final long byteOffset) {
    super(other, byteOffset);
  }
  
  public CudaPointer() {
    super();
  }
  
  public static CudaPointer to(float values[]) {
    return new CudaPointer(Pointer.to(values));
  }
  
  public static CudaPointer to(double values[]) {
    return new CudaPointer(Pointer.to(values));
  }
  
  @Override
  public CudaPointer withByteOffset(final long byteOffset) {
    return new CudaPointer(this, byteOffset);
  }
  
  @Override
  public long getByteOffset() {
    return super.getByteOffset();
  }
  
  
}
