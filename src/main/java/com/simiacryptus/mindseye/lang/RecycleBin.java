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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.*;
import java.util.function.Supplier;

import static com.simiacryptus.mindseye.lang.PersistanceMode.Weak;

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
    public void reset(@javax.annotation.Nonnull final double[] data, long size) {
      assert data.length == size;
      Arrays.fill(data, 0);
    }
  };
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(RecycleBin.class);
  private static volatile ScheduledExecutorService garbageTruck;
  
  private final StackCounter allocations = new StackCounter();
  private final StackCounter frees = new StackCounter();
  private final StackCounter recycle_put = new StackCounter();
  private final StackCounter recycle_get = new StackCounter();
  private final ConcurrentHashMap<Long, ConcurrentLinkedDeque<Supplier<T>>> recycling = new ConcurrentHashMap<>();
  private int profilingThreshold = 32 * 1024;
  private PersistanceMode persistanceMode = Weak;
  private int minLengthPerBuffer = 256;
  private double maxLengthPerBuffer = 1e9;
  private int maxItemsPerBuffer = 100;
  
  /**
   * Instantiates a new Recycle bin.
   */
  protected RecycleBin() {
    super();
    synchronized (recycling) {
      RecycleBin.getGarbageTruck().scheduleAtFixedRate(() -> recycling.forEach((k, v) -> {
        Supplier<T> poll;
        while (null != (poll = v.poll())) {
          T obj = poll.get();
          if (obj != null) {
            free2(obj, k);
          }
        }
      }), 10, 10, TimeUnit.SECONDS);
    }
  }
  
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
  public static boolean equals(@Nullable Object a, @Nullable Object b) {
    if (a == b) return true;
    if (a == null || b == null) return false;
    return a.equals(b);
  }
  
  /**
   * Free.
   *
   * @param obj  the obj
   * @param size the size
   */
  protected void free2(T obj, long size) {
    @Nullable StackCounter stackCounter = getFrees(size);
    if (null != stackCounter) {
      stackCounter.increment(size);
    }
    free(obj);
  }
  
  /**
   * Free.
   *
   * @param obj the obj
   */
  protected abstract void free(T obj);
  
  /**
   * Clear.
   */
  public void clear() {
    synchronized (recycling) {
      recycling.entrySet().stream().forEach(e -> e.getValue().forEach(x -> free2(x.get(), e.getKey())));
      recycling.clear();
    }
  }
  
  /**
   * Copy of double [ ].
   *
   * @param original the original
   * @param size     the size
   * @return the double [ ]
   */
  @Nullable
  public T copyOf(@Nullable final T original, long size) {
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
  @javax.annotation.Nonnull
  public abstract T create(long length);
  
  /**
   * Gets allocations.
   *
   * @param length the length
   * @return the allocations
   */
  @Nullable
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
  @Nullable
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
  @Nullable
  public StackCounter getRecycle_put(final long length) {
    if (!isProfiling(length)) return null;
    return recycle_put;
  }
  
  /**
   * Gets recycle submit.
   *
   * @param length the length
   * @return the recycle submit
   */
  @Nullable
  public StackCounter getRecycle_get(final long length) {
    if (!isProfiling(length)) return null;
    return recycle_get;
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
  public void printAllProfiling(@javax.annotation.Nonnull final PrintStream out) {
    printDetailedProfiling(out);
    printNetProfiling(out);
  }
  
  /**
   * Print detailed profiling.
   *
   * @param out the out
   */
  public void printDetailedProfiling(@javax.annotation.Nonnull final PrintStream out) {
    if (null != allocations) {
      out.println("Memory Allocation Profiling:\n\t" + allocations.toString().replaceAll("\n", "\n\t"));
    }
    if (null != frees) {
      out.println("Freed Memory Profiling:\n\t" + frees.toString().replaceAll("\n", "\n\t"));
    }
    if (null != recycle_put) {
      out.println("Recycle Bin (Put) Profiling:\n\t" + recycle_put.toString().replaceAll("\n", "\n\t"));
    }
    if (null != recycle_get) {
      out.println("Recycle Bin (Get) Profiling:\n\t" + recycle_get.toString().replaceAll("\n", "\n\t"));
    }
  }
  
  /**
   * Print net profiling.
   *
   * @param out the out
   */
  public void printNetProfiling(@Nullable final PrintStream out) {
    if (null != out && null != recycle_put && null != recycle_get) {
      out.println("Recycle Bin (Net) Profiling:\n\t" +
        StackCounter.toString(recycle_put, recycle_get, (a, b) -> a.getSum() - b.getSum())
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
    final ConcurrentLinkedDeque<Supplier<T>> bin;
    synchronized (recycling) {
      bin = recycling.get(length);
  
    }
    @Nullable StackCounter stackCounter = getRecycle_get(length);
    if (null != stackCounter) {
      stackCounter.increment(length);
    }
    if (null != bin) {
      final Supplier<T> ref = bin.poll();
      if (null != ref) {
        final T data = ref.get();
        if (null != data) {
          reset(data, length);
          return data;
        }
      }
    }
    return create(length, 1);
  }
  
  /**
   * Create t.
   *
   * @param length  the length
   * @param retries the retries
   * @return the t
   */
  @javax.annotation.Nonnull
  public T create(long length, int retries) {
    try {
      @javax.annotation.Nonnull T result = create(length);
      @Nullable StackCounter stackCounter = getAllocations(length);
      if (null != stackCounter) {
        stackCounter.increment(length);
      }
      return result;
    } catch (@javax.annotation.Nonnull final java.lang.OutOfMemoryError e) {
      if (retries <= 0) throw e;
    }
    clearMemory(length);
    return create(length, retries - 1);
  }
  
  private void clearMemory(long length) {
    long max = Runtime.getRuntime().maxMemory();
    long previous = Runtime.getRuntime().freeMemory();
    long size = getSize();
    logger.warn(String.format("Allocation of length %d failed; %s/%s used memory; %s items in recycle buffer; Clearing memory", length, previous, max, size));
    clear();
    System.gc();
    long after = Runtime.getRuntime().freeMemory();
    logger.warn(String.format("Clearing memory freed %s/%s bytes", previous - after, max));
  }
  
  /**
   * Gets size.
   *
   * @return the size
   */
  public long getSize() {
    return this.recycling.entrySet().stream().mapToLong(e -> e.getKey() * e.getValue().size()).sum();
  }
  
  /**
   * Recycle.
   *
   * @param data the data
   * @param size the size
   */
  public void recycle(@Nullable final T data, long size) {
    if (null != data && size >= getMinLengthPerBuffer() && size <= getMaxLengthPerBuffer()) {
      @Nullable StackCounter stackCounter = getRecycle_put(size);
      if (null != stackCounter) {
        stackCounter.increment(size);
      }
      ConcurrentLinkedDeque<Supplier<T>> bin = getBin(size);
      if (bin.size() < Math.min(Math.max(1, (int) (getMaxLengthPerBuffer() / size)), getMaxItemsPerBuffer())) {
        synchronized (bin) {
          if (!bin.stream().filter(x -> equals(x.get(), data)).findAny().isPresent()) {
            bin.add(wrap(data));
            return;
          }
        }
      }
    }
    free2(data, size);
  }
  
  /**
   * Want boolean.
   *
   * @param size the size
   * @return the boolean
   */
  public boolean want(long size) {
    if (size < getMinLengthPerBuffer()) return false;
    if (size > getMaxLengthPerBuffer()) return false;
    @Nullable StackCounter stackCounter = getRecycle_put(size);
    if (null != stackCounter) {
      stackCounter.increment(size);
    }
    ConcurrentLinkedDeque<Supplier<T>> bin = getBin(size);
    return bin.size() < Math.min(Math.max(1, (int) (getMaxLengthPerBuffer() / size)), getMaxItemsPerBuffer());
  }
  
  /**
   * Gets bin.
   *
   * @param size the size
   * @return the bin
   */
  protected ConcurrentLinkedDeque<Supplier<T>> getBin(long size) {
    ConcurrentLinkedDeque<Supplier<T>> bin;
    synchronized (recycling) {
      bin = recycling.get(size);
    }
    if (null == bin) {
      bin = new ConcurrentLinkedDeque<>();
      recycling.put(size, bin);
    }
    return bin;
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
   * @return the persistance mode
   */
  @javax.annotation.Nonnull
  public RecycleBin<T> setPersistanceMode(PersistanceMode persistanceMode) {
    this.persistanceMode = persistanceMode;
    return this;
  }
  
  /**
   * New ref reference.
   *
   * @param data the data
   * @return the reference
   */
  @javax.annotation.Nullable
  protected Supplier<T> wrap(T data) {
    return persistanceMode.wrap(data);
  }
  
  /**
   * Reset.
   *
   * @param data the data
   * @param size the size
   */
  public abstract void reset(T data, long size);
  
  /**
   * Sets profiling.
   *
   * @param threshold the threshold
   * @return the profiling
   */
  @javax.annotation.Nonnull
  public RecycleBin<T> setProfiling(final int threshold) {
    this.profilingThreshold = threshold;
    return this;
  }
  
  /**
   * Gets min bytes per buffer.
   *
   * @return the min bytes per buffer
   */
  public int getMinLengthPerBuffer() {
    return minLengthPerBuffer;
  }
  
  /**
   * Sets min bytes per buffer.
   *
   * @param minLengthPerBuffer the min bytes per buffer
   * @return the min bytes per buffer
   */
  @javax.annotation.Nonnull
  public RecycleBin<T> setMinLengthPerBuffer(int minLengthPerBuffer) {
    this.minLengthPerBuffer = minLengthPerBuffer;
    return this;
  }
  
  /**
   * Gets max bytes per buffer.
   *
   * @return the max bytes per buffer
   */
  public double getMaxLengthPerBuffer() {
    return maxLengthPerBuffer;
  }
  
  /**
   * Sets max bytes per buffer.
   *
   * @param maxLengthPerBuffer the max bytes per buffer
   * @return the max bytes per buffer
   */
  @javax.annotation.Nonnull
  public RecycleBin<T> setMaxLengthPerBuffer(double maxLengthPerBuffer) {
    this.maxLengthPerBuffer = maxLengthPerBuffer;
    return this;
  }
  
  /**
   * Gets max items per buffer.
   *
   * @return the max items per buffer
   */
  public int getMaxItemsPerBuffer() {
    return maxItemsPerBuffer;
  }
  
  /**
   * Sets max items per buffer.
   *
   * @param maxItemsPerBuffer the max items per buffer
   * @return the max items per buffer
   */
  @javax.annotation.Nonnull
  public RecycleBin<T> setMaxItemsPerBuffer(int maxItemsPerBuffer) {
    this.maxItemsPerBuffer = maxItemsPerBuffer;
    return this;
  }
  
}
