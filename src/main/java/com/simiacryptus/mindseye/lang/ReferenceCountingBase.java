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

package com.simiacryptus.mindseye.lang;

import com.simiacryptus.mindseye.lang.cudnn.CudaResource;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class ReferenceCountingBase implements ReferenceCounting {
  
  private final AtomicInteger references = new AtomicInteger(1);
  private final AtomicBoolean isFreed = new AtomicBoolean(false);
  private volatile StackTraceElement[] finalizedBy = null;
  
  @Override
  public void getRef() {
    if (references.incrementAndGet() <= 1) throw new IllegalStateException();
  }
  
  @Override
  public void freeRef() {
    int refs = references.decrementAndGet();
    if (refs < 0) {
      throw new IllegalStateException();
    }
    else if (refs == 0) {
      free();
    }
  }
  
  public boolean isFinalized() {
    return isFreed.get();
  }
  
  protected final void free() {
    assertAlive();
    assert references.get() == 0;
    if (!isFreed.getAndSet(true)) {
      finalizedBy = CudaResource.debugLifecycle ? Thread.currentThread().getStackTrace() : null;
      _free();
    }
  }
  
  public void assertAlive() {
    if (isFinalized())
      throw new IllegalStateException(null == finalizedBy ? "" : Arrays.stream(finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).orElse(""));
  }
  
  protected abstract void _free();
  
  @Override
  protected final void finalize() throws Throwable {
    if (!isFreed.getAndSet(true)) {
      finalizedBy = CudaResource.debugLifecycle ? Thread.currentThread().getStackTrace() : null;
      _free();
    }
  }
}
