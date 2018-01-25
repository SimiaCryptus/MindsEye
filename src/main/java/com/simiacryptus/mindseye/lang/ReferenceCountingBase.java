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

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Reference counting base.
 */
public abstract class ReferenceCountingBase implements ReferenceCounting {
  private static final Logger logger = LoggerFactory.getLogger(ReferenceCountingBase.class);
  private static final boolean DEBUG_LIFECYCLE = ReferenceCountingBase.class.desiredAssertionStatus();
  private static final boolean SUPPRESS_LOG = false;
  
  private final AtomicInteger references = new AtomicInteger(1);
  private final AtomicBoolean isFreed = new AtomicBoolean(false);
  private final StackTraceElement[] createdBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
  private final ArrayList<StackTraceElement[]> addRefs = new ArrayList<>();
  private final ArrayList<StackTraceElement[]> freeRefs = new ArrayList<>();
  private volatile StackTraceElement[] finalizedBy = null;
  private volatile boolean isFinalized = false;
  
  private static String getString(StackTraceElement[] trace) {
    return null == trace ? "?" : Arrays.stream(trace).map(x -> "at " + x).skip(2).reduce((a, b) -> a + "\n" + b).orElse("<Empty Stack>");
  }
  
  @Override
  public void addRef() {
    assertAlive();
    if (references.incrementAndGet() <= 1) throw new IllegalStateException();
    if (DEBUG_LIFECYCLE) addRefs.add(Thread.currentThread().getStackTrace());
  }
  
  @Override
  public void freeRef() {
    if (isFinalized) {
      //logger.debug("Object has been finalized");
      return;
    }
    int refs = references.decrementAndGet();
    if (refs < 0) {
      if (!SUPPRESS_LOG) {
        //SUPPRESS_LOG = true;
        logger.warn(String.format("Error freeing reference for %s", getClass().getSimpleName()));
        logger.warn(detailString());
      }
      throw new LifecycleException();
    }
    else if (refs == 0) {
      assert references.get() == 0;
      if (!isFreed.getAndSet(true)) {
        finalizedBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
        _free();
      }
    }
    else {
      if (DEBUG_LIFECYCLE) freeRefs.add(Thread.currentThread().getStackTrace());
    }
  }
  
  private String detailString() {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    PrintStream out = new PrintStream(buffer);
    out.print(String.format("Object %s ",
                            getClass().getSimpleName()));
    if (null != createdBy) {
      out.println(String.format("created by \n\t%s",
                                getString(createdBy).replaceAll("\n", "\n\t")));
    }
    for (StackTraceElement[] stack : addRefs) {
      out.println(String.format("reference added by \n\t%s",
                                getString(stack).replaceAll("\n", "\n\t")));
    }
    for (StackTraceElement[] stack : freeRefs) {
      out.println(String.format("reference removed by \n\t%s",
                                getString(stack).replaceAll("\n", "\n\t")));
    }
    if (null != finalizedBy) {
      out.println(String.format("freed by \n\t%s",
                                getString(this.finalizedBy).replaceAll("\n", "\n\t")));
    }
    out.println(String.format("with current stack \n\t%s",
                              getString(Thread.currentThread().getStackTrace()).replaceAll("\n", "\n\t")));
    out.close();
    return buffer.toString();
  }
  
  public final boolean isFinalized() {
    return isFreed.get();
  }
  
  /**
   * Assert alive.
   */
  public final void assertAlive() {
    if (isFinalized) {
      throw new LifecycleException("Object has been finalized");
    }
    if (isFinalized()) {
      if (!SUPPRESS_LOG) {
        //SUPPRESS_LOG = true;
        logger.warn(String.format("Using freed reference for %s", getClass().getSimpleName()));
        logger.warn(detailString());
      }
      throw new LifecycleException();
    }
  }
  
  /**
   * Free.
   */
  protected abstract void _free();
  
  @Override
  protected final void finalize() throws Throwable {
    isFinalized = true;
    if (!isFreed.getAndSet(true)) {
      finalizedBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
      _free();
    }
  }
  
}
