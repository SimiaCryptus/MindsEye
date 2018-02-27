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

import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The base implementation for ReferenceCounting objects. Provides state management and debugging facilities. If
 * assertions are enabled, stack traces are recorded to provide detailed logs for debugging LifecycleExceptions.
 */
public abstract class ReferenceCountingBase implements ReferenceCounting {
  
  private static final Logger logger = LoggerFactory.getLogger(ReferenceCountingBase.class);
  private static final long LOAD_TIME = System.nanoTime();
  private static final UUID jvmId = UUID.randomUUID();
  
  static {
    if (CoreSettings.INSTANCE == null) throw new RuntimeException();
  }

  private final UUID objectId = CoreSettings.INSTANCE.isLifecycleDebug() ? UUID.randomUUID() : jvmId;
  private final AtomicInteger references = new AtomicInteger(1);
  private final AtomicBoolean isFreed = new AtomicBoolean(false);
  @Nullable
  private final StackTraceElement[] createdBy = CoreSettings.INSTANCE.isLifecycleDebug() ? Thread.currentThread().getStackTrace() : null;
  private final LinkedList<StackTraceElement[]> addRefs = new LinkedList<>();
  private final LinkedList<StackTraceElement[]> freeRefs = new LinkedList<>();
  private final LinkedList<UUID> addRefObjs = new LinkedList<>();
  private final LinkedList<UUID> freeRefObjs = new LinkedList<>();
  private volatile boolean isFinalized = false;
  private boolean detached = false;
  
  @javax.annotation.Nonnull
  private static String getString(@Nullable StackTraceElement[] trace) {
    return null == trace ? "" : Arrays.stream(trace).map(x -> "at " + x).skip(2).reduce((a, b) -> a + "\n" + b).orElse("");
  }
  
  /**
   * Detail string string.
   *
   * @param obj           the obj
   * @param includeCaller the include caller
   * @return the string
   */
  public static String referenceReport(@javax.annotation.Nonnull ReferenceCountingBase obj, boolean includeCaller) {
    return obj.referenceReport(includeCaller, obj.isFinalized());
  }
  
  @Override
  public int currentRefCount() {
    return references.get();
  }
  
  @Override
  public void addRef() {
    addRef(this);
  }
  
  @Override
  public void claimRef(ReferenceCounting obj) {
    assertAlive();
    synchronized (addRefObjs) {
      for (Iterator<UUID> iterator = addRefObjs.iterator(); iterator.hasNext(); ) {
        final UUID addRefObj = iterator.next();
        if (addRefObj.equals(this.objectId)) {
          iterator.remove();
          addRefObjs.add(obj.getObjectId());
          return;
        }
      }
    }
    throw new IllegalStateException("No reference to claim found");
  }
  
  @Override
  public void addRef(ReferenceCounting obj) {
    assertAlive();
    if (references.incrementAndGet() <= 1) throw new IllegalStateException(referenceReport(true, isFinalized()));
    if (CoreSettings.INSTANCE.isLifecycleDebug()) {
      addRefs.add(Thread.currentThread().getStackTrace());
    }
    synchronized (addRefObjs) {
      addRefObjs.add(obj.getObjectId());
    }
  }
  
  public final boolean isFinalized() {
    return isFreed.get();
  }
  
  /**
   * Reference report string.
   *
   * @param includeCaller the include caller
   * @param isFinalized   the is finalized
   * @return the string
   */
  public String referenceReport(boolean includeCaller, boolean isFinalized) {
    @javax.annotation.Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    @javax.annotation.Nonnull PrintStream out = new PrintStream(buffer);
    out.print(String.format("Object %s %s (%d refs, %d frees) ",
      getClass().getName(), getObjectId().toString(), 1 + addRefObjs.size(), freeRefObjs.size()));
    if (null != createdBy) {
      out.println(String.format("created by \n\t%s",
        getString(createdBy).replaceAll("\n", "\n\t")));
    }
    synchronized (addRefObjs) {
      for (int i = 0; i < addRefObjs.size(); i++) {
        StackTraceElement[] stack = i < addRefs.size() ? addRefs.get(i) : new StackTraceElement[]{};
        UUID linkObj = addRefObjs.get(i);
        String linkStr = this.equals(linkObj) ? "" : linkObj.toString();
        out.println(String.format("reference added by %s\n\t%s", linkStr,
          getString(stack).replaceAll("\n", "\n\t")));
      }
    }
    synchronized (freeRefObjs) {
      for (int i = 0; i < freeRefObjs.size() - (isFinalized ? 1 : 0); i++) {
        StackTraceElement[] stack = i < freeRefs.size() ? freeRefs.get(i) : new StackTraceElement[]{};
        UUID linkObj = freeRefObjs.get(i);
        String linkStr = objectId == linkObj ? "" : linkObj.toString();
        out.println(String.format("reference removed by %s\n\t%s", linkStr,
          getString(stack).replaceAll("\n", "\n\t")));
      }
      if (isFinalized) {
        UUID linkObj = freeRefObjs.isEmpty() ? objectId : freeRefObjs.get(freeRefObjs.size() - 1);
        String linkStr = objectId.equals(linkObj) ? "" : linkObj.toString();
        out.println(String.format("freed by %s\n\t%s", linkStr,
          (0 == freeRefs.size() ? "" : getString(freeRefs.get(freeRefs.size() - 1))).replaceAll("\n", "\n\t")));
      }
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
      logger.warn(String.format("Using freed reference for %s", getClass().getSimpleName()));
      logger.warn(referenceReport(true, isFinalized()));
      throw new LifecycleException(this);
    }
  }
  
  @Override
  public void freeRef() {
    freeRef(this);
  }
  
  @Override
  public void freeRef(ReferenceCounting obj) {
    if (isFinalized) {
      //logger.debug("Object has been finalized");
      return;
    }
    int refs = references.decrementAndGet();
    if (refs < 0) {
      logger.warn(String.format("Error freeing reference for %s", getClass().getSimpleName()));
      logger.warn(referenceReport(true, isFinalized()));
      throw new LifecycleException(this);
    }
  
    synchronized (freeRefObjs) {
      if (CoreSettings.INSTANCE.isLifecycleDebug()) freeRefs.add(Thread.currentThread().getStackTrace());
      freeRefObjs.add(obj.getObjectId());
    }
    if (refs == 0) {
      assert references.get() == 0;
      if (!isFreed.getAndSet(true)) {
        try {
          _free();
        } catch (LifecycleException e) {
          logger.info("Error freeing resources: " + referenceReport(true, isFinalized()));
          throw e;
        }
      }
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
      if (!isDetached()) {
        if (logger.isDebugEnabled()) {
          logger.debug(String.format("Instance Reclaimed by GC at %.9f: %s", (System.nanoTime() - LOAD_TIME) / 1e9, referenceReport(false, false)));
        }
      }
      synchronized (freeRefObjs) {
        if (CoreSettings.INSTANCE.isLifecycleDebug()) freeRefs.add(Thread.currentThread().getStackTrace());
        freeRefObjs.add(this.objectId);
      }
      _free();
    }
  }
  
  /**
   * Is floating boolean.
   *
   * @return the boolean
   */
  public boolean isDetached() {
    return detached;
  }
  
  /**
   * Sets floating.
   */
  public void detach() {
    this.detached = true;
  }
  
  @Override
  public UUID getObjectId() {
    return objectId;
  }
}
