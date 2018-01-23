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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Reference counting base.
 */
public abstract class ReferenceCountingBase implements ReferenceCounting {
  private static final Logger logger = LoggerFactory.getLogger(ReferenceCountingBase.class);
  
  /**
   * The constant debugLifecycle.
   */
  public static boolean debugLifecycle = true;
  /**
   * The Created by.
   */
  protected final StackTraceElement[] createdBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
  private final AtomicInteger references = new AtomicInteger(1);
  private final AtomicBoolean isFreed = new AtomicBoolean(false);
  private volatile StackTraceElement[] finalizedBy = null;
  
  @Override
  public void addRef() {
    assertAlive();
    if (references.incrementAndGet() <= 1) throw new IllegalStateException();
  }
  
  @Override
  public void freeRef() {
    int refs = references.decrementAndGet();
    if (refs < 0) {
      String createdByStr = null == finalizedBy ? "?" : Arrays.stream(createdBy).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      String finalizedByStr = null == finalizedBy ? "?" : Arrays.stream(finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      String currentStackStr = Arrays.stream(Thread.currentThread().getStackTrace()).skip(2).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      logger.warn(String.format("Error freeing reference for %s created by \n\t%s\n; freed by \n\t%s\n; with current stack \n\t%s",
                                getClass().getSimpleName(),
                                createdByStr.replaceAll("\n", "\n\t"),
                                finalizedByStr.replaceAll("\n", "\n\t"),
                                currentStackStr.replaceAll("\n", "\n\t")));
      throw new IllegalStateException();
    }
    else if (refs == 0) {
      assert references.get() == 0;
      if (!isFreed.getAndSet(true)) {
        finalizedBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
        _free();
      }
    }
  }
  
  public boolean isFinalized() {
    return isFreed.get();
  }
  
  /**
   * Assert alive.
   */
  public void assertAlive() {
    if (isFinalized()) {
      String createdByStr = null == finalizedBy ? "?" : Arrays.stream(createdBy).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      String finalizedByStr = null == finalizedBy ? "?" : Arrays.stream(finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      String currentStackStr = Arrays.stream(Thread.currentThread().getStackTrace()).skip(2).map(x -> x.toString()).reduce((a, b) -> a + "\n" + b).orElse("?");
      logger.warn(String.format("Using freed reference for %s created by \n\t%s\n; freed by \n\t%s\n; with current stack \n\t%s",
                                getClass().getSimpleName(),
                                createdByStr.replaceAll("\n", "\n\t"),
                                finalizedByStr.replaceAll("\n", "\n\t"),
                                currentStackStr.replaceAll("\n", "\n\t")));
      throw new IllegalStateException();
    }
  }
  
  /**
   * Free.
   */
  protected abstract void _free();
  
  @Override
  protected final void finalize() throws Throwable {
    if (!isFreed.getAndSet(true)) {
      finalizedBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
      _free();
    }
  }
}
