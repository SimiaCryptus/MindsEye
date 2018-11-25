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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.*;
import java.util.function.Supplier;

import static com.simiacryptus.mindseye.lang.PersistanceMode.WEAK;

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
    protected void free(double[] obj) {
    }

    @Override
    public double[] create(final long length) {
      return new double[(int) length];
    }

    @Override
    public void reset(@Nonnull final double[] data, long size) {
      assert data.length == size;
      Arrays.fill(data, 0);
    }
  }.setPersistanceMode(CoreSettings.INSTANCE().getDoubleCacheMode());
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(RecycleBin.class);
  private static volatile ScheduledExecutorService garbageTruck;
  private final Map<Long, ConcurrentLinkedDeque<ObjectWrapper>> buckets = new ConcurrentHashMap<>();
  private final StackCounter allocations = new StackCounter();
  private final StackCounter frees = new StackCounter();
  private final StackCounter recycle_put = new StackCounter();
  private final StackCounter recycle_get = new StackCounter();
  private int purgeFreq;
  private int profilingThreshold = Integer.MAX_VALUE;
  private PersistanceMode persistanceMode = WEAK;
  private int minLengthPerBuffer = 256;
  private double maxLengthPerBuffer = 1e9;
  private int maxItemsPerBuffer = 100;

  /**
   * Instantiates a new Recycle bin.
   */
  protected RecycleBin() {
    super();
    purgeFreq = 10;
    RecycleBin.getGarbageTruck().scheduleAtFixedRate(() -> {
      buckets.forEach((k, v) -> {
        ObjectWrapper poll;
        ArrayList<ObjectWrapper> young = new ArrayList<>();
        while (null != (poll = v.poll())) {
          if (poll.age() > purgeFreq) {
            T obj = poll.obj.get();
            if (obj != null) {
              freeItem(obj, k);
            }
          } else {
            young.add(poll);
          }
        }
        v.addAll(young);
      });
    }, purgeFreq, purgeFreq, TimeUnit.SECONDS);
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
   * Clear.
   *
   * @return the long
   */
  public long clear() {
    return buckets.keySet().stream().mapToLong(length -> buckets.remove(length).stream().mapToLong(ref -> {
      return freeItem(ref.obj.get(), length);
    }).sum()).sum();
  }

  /**
   * Free.
   *
   * @param obj  the obj
   * @param size the size
   * @return the long
   */
  protected long freeItem(T obj, long size) {
    @Nullable StackCounter stackCounter = getFrees(size);
    if (null != stackCounter) {
      stackCounter.increment(size);
    }
    if (null != obj) free(obj);
    return size;
  }

  /**
   * Free.
   *
   * @param obj the obj
   */
  protected abstract void free(T obj);

  /**
   * Obtain double [ ].
   *
   * @param length the length
   * @return the double [ ]
   */
  public T obtain(final long length) {
    final ConcurrentLinkedDeque<ObjectWrapper> bin = buckets.get(length);
    @Nullable StackCounter stackCounter = getRecycle_get(length);
    if (null != stackCounter) {
      stackCounter.increment(length);
    }
    if (null != bin) {
      final ObjectWrapper ref = bin.poll();
      if (null != ref) {
        final T data = ref.obj.get();
        if (null != data) {
          reset(data, length);
          return data;
        }
      }
    }
    return create(length, 1);
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
  @Nonnull
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
  public void printAllProfiling(@Nonnull final PrintStream out) {
    printDetailedProfiling(out);
    printNetProfiling(out);
  }

  /**
   * Print detailed profiling.
   *
   * @param out the out
   */
  public void printDetailedProfiling(@Nonnull final PrintStream out) {
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
      ConcurrentLinkedDeque<ObjectWrapper> bin = getBin(size);
      if (bin.size() < Math.min(Math.max(1, (int) (getMaxLengthPerBuffer() / size)), getMaxItemsPerBuffer())) {
        synchronized (bin) {
          if (!bin.stream().filter(x -> equals(x.obj.get(), data)).findAny().isPresent()) {
            bin.add(new ObjectWrapper(wrap(data)));
            return;
          }
        }
      }
    }
    freeItem(data, size);
  }

  /**
   * Create t.
   *
   * @param length  the length
   * @param retries the retries
   * @return the t
   */
  @Nonnull
  public T create(long length, int retries) {
    try {
      @Nonnull T result = create(length);
      @Nullable StackCounter stackCounter = getAllocations(length);
      if (null != stackCounter) {
        stackCounter.increment(length);
      }
      return result;
    } catch (@Nonnull final OutOfMemoryError e) {
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
    return this.buckets.entrySet().stream().mapToLong(e -> e.getKey() * e.getValue().size()).sum();
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
    ConcurrentLinkedDeque<ObjectWrapper> bin = getBin(size);
    return bin.size() < Math.min(Math.max(1, (int) (getMaxLengthPerBuffer() / size)), getMaxItemsPerBuffer());
  }

  /**
   * Gets bin.
   *
   * @param size the size
   * @return the bin
   */
  protected ConcurrentLinkedDeque<ObjectWrapper> getBin(long size) {
    return buckets.computeIfAbsent(size, x -> new ConcurrentLinkedDeque<>());
  }

  /**
   * Gets purge freq.
   *
   * @return the purge freq
   */
  public int getPurgeFreq() {
    return purgeFreq;
  }

  /**
   * Sets purge freq.
   *
   * @param purgeFreq the purge freq
   * @return the purge freq
   */
  public RecycleBin<T> setPurgeFreq(int purgeFreq) {
    this.purgeFreq = purgeFreq;
    return this;
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
  @Nonnull
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
  @Nullable
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
  @Nonnull
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
  @Nonnull
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
  @Nonnull
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
  @Nonnull
  public RecycleBin<T> setMaxItemsPerBuffer(int maxItemsPerBuffer) {
    this.maxItemsPerBuffer = maxItemsPerBuffer;
    return this;
  }

  private class ObjectWrapper {
    /**
     * The Obj.
     */
    public final Supplier<T> obj;
    /**
     * The Created at.
     */
    public final long createdAt = System.nanoTime();

    private ObjectWrapper(final Supplier<T> obj) {
      this.obj = obj;
    }

    /**
     * Age double.
     *
     * @return the double
     */
    public final double age() {
      return (System.nanoTime() - createdAt) / 1e9;
    }
  }
}
