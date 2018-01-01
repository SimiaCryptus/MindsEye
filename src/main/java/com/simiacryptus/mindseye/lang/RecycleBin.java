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

package com.simiacryptus.mindseye.lang;

import com.google.common.util.concurrent.ThreadFactoryBuilder;

import java.io.PrintStream;
import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.concurrent.*;

import static com.simiacryptus.mindseye.lang.RecycleBin.PersistanceMode.Soft;

/**
 * This is a recycling mechanism to reuse short-term-lifecycle T objects of regular length. It is a convenience
 * mechanism to optimize tight loops which would otherwise require careful and complex coding to minimize allocations
 * and avoid excessive GC load
 *
 * @param <T> the type parameter
 */
public abstract class RecycleBin<T> {
  
  
  /**
   * The constant DOUBLES.
   */
  public static final RecycleBin<double[]> DOUBLES = new RecycleBin<double[]>() {
    @Override
    protected void free(double[] obj) { }
  
    @Override
    public double[] create(final long length) {
      return new double[(int) length];
    }
  
    @Override
    public void reset(final double[] data, long size) {
      Arrays.fill(data, 0);
    }
  };
  private static volatile ScheduledExecutorService garbageTruck;
  
  static {
    if (RecycleBin.class.desiredAssertionStatus()) {
      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        RecycleBin.DOUBLES.printNetProfiling(System.err);
      }));
    }
  }
  
  private final StackCounter allocations = new StackCounter();
  private final StackCounter frees = new StackCounter();
  private final StackCounter recycle_get = new StackCounter();
  private final StackCounter recycle_submit = new StackCounter();
  private final ConcurrentHashMap<Long, ConcurrentLinkedDeque<Reference<T>>> recycling = new ConcurrentHashMap<>();
  private int profilingThreshold = 32 * 1024;
  private PersistanceMode persistanceMode = Soft;
  
  protected RecycleBin() {
    super();
    RecycleBin.getGarbageTruck().scheduleAtFixedRate(() -> recycling.forEach((k, v) -> {
      final StackCounter stackCounter = getRecycle_submit(k.intValue());
      if (null != stackCounter) {
        Reference<T> poll;
        while (null != (poll = v.poll())) {
          T obj = poll.get();
          if (obj != null) {
            stackCounter.increment(k);
            free(obj);
          }
        }
      }
      else {
        v.clear();
      }
    }), 10, 10, TimeUnit.SECONDS);
  }
  
  protected abstract void free(T obj);
  
  /**
   * Gets garbage truck.
   *
   * @return the garbage truck
   */
  public static ScheduledExecutorService getGarbageTruck() {
    if (null == RecycleBin.garbageTruck) {
      synchronized (RecycleBin.class) {
        if (null == RecycleBin.garbageTruck) {
          RecycleBin.garbageTruck = Executors.newScheduledThreadPool(1, new ThreadFactoryBuilder().setDaemon(true).build());
        }
      }
    }
    return RecycleBin.garbageTruck;
  }
  
  /**
   * Equals boolean.
   *
   * @param a the a
   * @param b the b
   * @return the boolean
   */
  public static boolean equals(Object a, Object b) {
    if (a == b) return true;
    if (a == null || b == null) return false;
    return a.equals(b);
  }
  
  /**
   * Clear.
   */
  public synchronized void clear() {
    recycling.values().stream().flatMap(x -> x.stream()).forEach(x -> free(x.get()));
    recycling.clear();
  }
  
  /**
   * Copy of double [ ].
   *
   * @param original the original
   * @param size
   * @return the double [ ]
   */
  public T copyOf(final T original, long size) {
    if (null == original) return null;
    final T copy = obtain(size);
    System.arraycopy(original, 0, copy, 0, (int) size);
    return copy;
  }
  
  /**
   * Create t.
   *
   * @param length the length
   * @return the t
   */
  public abstract T create(long length);
  
  /**
   * Gets allocations.
   *
   * @param length the length
   * @return the allocations
   */
  public StackCounter getAllocations(final long length) {
    if (!isProfiling(length)) return null;
    return allocations;
  }
  
  /**
   * Gets frees.
   *
   * @param length the length
   * @return the frees
   */
  public StackCounter getFrees(final long length) {
    if (!isProfiling(length)) return null;
    return frees;
  }
  
  /**
   * Gets recycle get.
   *
   * @param length the length
   * @return the recycle get
   */
  public StackCounter getRecycle_get(final long length) {
    if (!isProfiling(length)) return null;
    return recycle_get;
  }
  
  /**
   * Gets recycle submit.
   *
   * @param length the length
   * @return the recycle submit
   */
  public StackCounter getRecycle_submit(final long length) {
    if (!isProfiling(length)) return null;
    return recycle_submit;
  }
  
  /**
   * Is profiling boolean.
   *
   * @param length the length
   * @return the boolean
   */
  public boolean isProfiling(final long length) {
    return length > profilingThreshold;
  }
  
  /**
   * Print all profiling.
   *
   * @param out the out
   */
  public void printAllProfiling(final PrintStream out) {
    printDetailedProfiling(out);
    printNetProfiling(out);
  }
  
  /**
   * Print detailed profiling.
   *
   * @param out the out
   */
  public void printDetailedProfiling(final PrintStream out) {
    if (null != allocations) {
      out.println("Memory Allocation Profiling:\n\t" + allocations.toString().replaceAll("\n", "\n\t"));
    }
    if (null != frees) {
      out.println("Freed Memory Profiling:\n\t" + frees.toString().replaceAll("\n", "\n\t"));
    }
    if (null != recycle_get) {
      out.println("Recycle Bin (Put) Profiling:\n\t" + recycle_get.toString().replaceAll("\n", "\n\t"));
    }
    if (null != recycle_submit) {
      out.println("Recycle Bin (Get) Profiling:\n\t" + recycle_submit.toString().replaceAll("\n", "\n\t"));
    }
  }
  
  /**
   * Print net profiling.
   *
   * @param out the out
   */
  public void printNetProfiling(final PrintStream out) {
    if (null != out && null != recycle_get && null != recycle_submit) {
      out.println("Recycle Bin (Net) Profiling:\n\t" +
                    StackCounter.toString(recycle_get, recycle_submit, (a, b) -> a.getSum() - b.getSum())
                                .replaceAll("\n", "\n\t"));
    }
  }
  
  /**
   * Obtain double [ ].
   *
   * @param length the length
   * @return the double [ ]
   */
  public T obtain(final long length) {
    final ConcurrentLinkedDeque<Reference<T>> bin = recycling.get(length);
    StackCounter stackCounter = getRecycle_get(length);
    if (null != stackCounter) {
      stackCounter.increment(length);
    }
    if (null != bin) {
      final Reference<T> ref = bin.poll();
      if (null != ref) {
        final T data = ref.get();
        if (null != data) {
          reset(data, length);
          return data;
        }
      }
    }
    try {
      stackCounter = getAllocations(length);
      if (null != stackCounter) {
        stackCounter.increment(length);
      }
      return create(length);
    } catch (final java.lang.OutOfMemoryError e) {
      try {
        clear();
        System.gc();
        return create(length);
      } catch (final java.lang.OutOfMemoryError e2) {
        throw new OutOfMemoryError("Could not allocate " + length + " bytes", e2);
      }
    }
  }
  
  /**
   * Recycle.
   *
   * @param data the data
   * @param size
   */
  public void recycle(final T data, long size) {
    if (null == data) return;
    if (size < 256) return;
    ConcurrentLinkedDeque<Reference<T>> bin = recycling.get(size);
    if (null == bin) {
      bin = new ConcurrentLinkedDeque<>();
      recycling.put(size, bin);
    }
    StackCounter stackCounter = getRecycle_submit(size);
    if (null != stackCounter) {
      stackCounter.increment(size);
    }
    if (bin.size() < Math.max(1, (int) (1e8 / size))) {
      synchronized (bin) {
        if (!bin.stream().filter(x -> equals(x.get(), data)).findAny().isPresent()) {
          bin.add(newRef(data));
        }
      }
    }
    else {
      stackCounter = getFrees(size);
      if (null != stackCounter) {
        stackCounter.increment(size);
      }
    }
  }
  
  /**
   * Gets persistance mode.
   *
   * @return the persistance mode
   */
  public PersistanceMode getPersistanceMode() {
    return persistanceMode;
  }
  
  /**
   * Sets persistance mode.
   *
   * @param persistanceMode the persistance mode
   */
  public void setPersistanceMode(PersistanceMode persistanceMode) {
    this.persistanceMode = persistanceMode;
  }
  
  /**
   * New ref reference.
   *
   * @param data the data
   * @return the reference
   */
  protected Reference<T> newRef(T data) {
    switch (persistanceMode) {
      case Soft:
        return new SoftReference<T>(data);
      case Weak:
        return new WeakReference<T>(data);
      case Phantom:
        return new PhantomReference<T>(data, null);
      default:
        throw new IllegalStateException(persistanceMode.toString());
    }
  }
  
  /**
   * Reset.
   *
   * @param data the data
   * @param size
   */
  public abstract void reset(T data, long size);
  
  /**
   * Sets profiling.
   *
   * @param threshold the threshold
   * @return the profiling
   */
  public RecycleBin<T> setProfiling(final int threshold) {
    this.profilingThreshold = threshold;
    return this;
  }
  
  /**
   * The enum Persistance mode.
   */
  public enum PersistanceMode {
    /**
     * Soft persistance mode.
     */
    Soft,
    /**
     * Weak persistance mode.
     */
    Weak,
    /**
     * Phantom persistance mode.
     */
    Phantom
  }
}
