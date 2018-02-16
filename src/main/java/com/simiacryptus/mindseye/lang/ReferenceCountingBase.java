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

import com.simiacryptus.mindseye.test.TestUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The base implementation for ReferenceCounting objects. Provides state management and debugging facilities. If
 * assertions are enabled, stack traces are recorded to provide detailed logs for debugging LifecycleExceptions.
 */
public abstract class ReferenceCountingBase implements ReferenceCounting {
  private static final Logger logger = LoggerFactory.getLogger(ReferenceCountingBase.class);
  private static final boolean DEBUG_LIFECYCLE = Boolean.parseBoolean(System.getProperty("DEBUG_LIFECYCLE", Boolean.toString(TestUtil.CONSERVATIVE)));
  private static final boolean SUPPRESS_LOG = false;
  private static final long LOAD_TIME = System.nanoTime();
  private final AtomicInteger references = new AtomicInteger(1);
  private final AtomicBoolean isFreed = new AtomicBoolean(false);
  @Nullable
  private final StackTraceElement[] createdBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
  private final ConcurrentLinkedDeque<StackTraceElement[]> addRefs = new ConcurrentLinkedDeque<>();
  private final ConcurrentLinkedDeque<StackTraceElement[]> freeRefs = new ConcurrentLinkedDeque<>();
  @Nullable
  private volatile StackTraceElement[] finalizedBy = null;
  private volatile boolean isFinalized = false;
  private boolean floating = false;
  
  @javax.annotation.Nonnull
  private static String getString(@Nullable StackTraceElement[] trace) {
    return null == trace ? "?" : Arrays.stream(trace).map(x -> "at " + x).skip(2).reduce((a, b) -> a + "\n" + b).orElse("<Empty Stack>");
  }
  
  /**
   * Detail string string.
   *
   * @param obj           the obj
   * @param includeCaller the include caller
   * @return the string
   */
  public static String detailString(@javax.annotation.Nonnull ReferenceCountingBase obj, boolean includeCaller) {
    return obj.detailString(includeCaller);
  }
  
  @Override
  public int currentRefCount() {
    return references.get();
  }
  
  @Override
  public void addRef() {
    assertAlive();
    if (references.incrementAndGet() <= 1) throw new IllegalStateException(detailString(true));
    if (DEBUG_LIFECYCLE) addRefs.add(Thread.currentThread().getStackTrace());
  }
  
  public final boolean isFinalized() {
    return isFreed.get();
  }
  
  private String detailString(boolean includeCaller) {
    @javax.annotation.Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    @javax.annotation.Nonnull PrintStream out = new PrintStream(buffer);
    out.print(String.format("Object %s (%d refs, %d frees) ",
      getClass().getName(), 1 + addRefs.size(), freeRefs.size()));
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
    if (includeCaller) out.println(String.format("with current stack \n\t%s",
      getString(Thread.currentThread().getStackTrace()).replaceAll("\n", "\n\t")));
    out.close();
    return buffer.toString();
  }
  
  /**
   * Assert alive.
   */
  public final void assertAlive() {
    if (isFinalized) {
      throw new LifecycleException(this);
    }
    if (isFinalized()) {
      if (!SUPPRESS_LOG) {
        //SUPPRESS_LOG = true;
        logger.warn(String.format("Using freed reference for %s", getClass().getSimpleName()));
        logger.warn(detailString(true));
      }
      throw new LifecycleException(this);
    }
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
        logger.warn(detailString(true));
      }
      throw new LifecycleException(this);
    }
    else if (refs == 0) {
      assert references.get() == 0;
      if (!isFreed.getAndSet(true)) {
        finalizedBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
        try {
          _free();
        } catch (LifecycleException e) {
          logger.info("Error freeing resources: " + detailString(true));
          throw e;
        }
      }
    }
    else {
      if (DEBUG_LIFECYCLE) freeRefs.add(Thread.currentThread().getStackTrace());
    }
  }
  
  /**
   * Free.
   */
  protected void _free() {}
  
  @Override
  protected final void finalize() throws Throwable {
    isFinalized = true;
    if (!isFreed.getAndSet(true)) {
      if (!isFloating()) {
        if (DEBUG_LIFECYCLE && logger.isDebugEnabled()) {
          logger.debug(String.format("Instance Reclaimed by GC at %.9f: %s", (System.nanoTime() - LOAD_TIME) / 1e9, detailString(false)));
        }
      }
      finalizedBy = DEBUG_LIFECYCLE ? Thread.currentThread().getStackTrace() : null;
      _free();
    }
  }
  
  /**
   * Is floating boolean.
   *
   * @return the boolean
   */
  public boolean isFloating() {
    return floating;
  }
  
  /**
   * Sets floating.
   *
   * @param floating the floating
   */
  public void setFloating(boolean floating) {
    this.floating = floating;
  }
}
