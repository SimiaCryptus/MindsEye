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

import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.*;

/**
 * This is a recycling mechanism to reuse short-term-lifecycle T objects of regular length.
 * It is a convenience mechanism to optimize tight loops which
 * would otherwise require careful and complex coding to minimize allocations and avoid excessive GC load
 *
 * @param <T> the type parameter
 */
public abstract class RecycleBin<T> {
  
  
  /**
   * The constant DOUBLES.
   */
  public static final RecycleBin<double[]> DOUBLES = new RecycleBin<double[]>() {
    @Override
    protected int length(double[] data) {
      return data.length;
    }
    
    @Override
    public double[] create(int length) {
      return new double[length];
    }
    
    @Override
    public void reset(double[] data) {
      Arrays.fill(data, 0);
    }
  };
  private static volatile ScheduledExecutorService garbageTruck;
  
  private final ConcurrentHashMap<Integer, ConcurrentLinkedDeque<T>> recycling = new ConcurrentHashMap<>();
  private final StackCounter allocations = new StackCounter();
  private final StackCounter recycle_submit = new StackCounter();
  private final StackCounter recycle_get = new StackCounter();
  private final StackCounter frees = new StackCounter();
  private int profilingThreshold = 32 * 1024;
  
  static {
    if (RecycleBin.class.desiredAssertionStatus()) {
      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        DOUBLES.printNetProfiling(System.err);
      }));
    }
  }
  
  private RecycleBin() {
    super();
    getGarbageTruck().scheduleAtFixedRate(new Runnable() {
      @Override
      public void run() {
        recycling.forEach((k, v) -> {
          StackCounter stackCounter = getRecycle_submit(k.intValue());
          if (null != stackCounter) {
            T poll;
            while (null != (poll = v.poll())) {
              stackCounter.increment(length(poll));
            }
          }
          else {
            v.clear();
          }
        });
      }
    }, 10, 10, TimeUnit.SECONDS);
  }
  
  /**
   * Gets garbage truck.
   *
   * @return the garbage truck
   */
  public static ScheduledExecutorService getGarbageTruck() {
    if (null == garbageTruck) {
      synchronized (RecycleBin.class) {
        if (null == garbageTruck) {
          garbageTruck = Executors.newScheduledThreadPool(1);
        }
      }
    }
    return garbageTruck;
  }
  
  /**
   * Length int.
   *
   * @param data the data
   * @return the int
   */
  protected abstract int length(T data);
  
  /**
   * Print all profiling.
   *
   * @param out the out
   */
  public void printAllProfiling(PrintStream out) {
    printDetailedProfiling(out);
    printNetProfiling(out);
  }
  
  /**
   * Print detailed profiling.
   *
   * @param out the out
   */
  public void printDetailedProfiling(PrintStream out) {
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
  public void printNetProfiling(PrintStream out) {
    if (null != out && null != recycle_get && null != recycle_submit) {
      out.println("Recycle Bin (Net) Profiling:\n\t" +
        StackCounter.toString(recycle_get, recycle_submit, (a, b) -> a.getSum() - b.getSum())
          .replaceAll("\n", "\n\t"));
    }
  }
  
  /**
   * Recycle.
   *
   * @param data the data
   */
  public void recycle(T data) {
    if (null == data) return;
    int length = length(data);
    if (length < 256) return;
    ConcurrentLinkedDeque<T> bin = recycling.get(length);
    if (null == bin) {
      bin = new ConcurrentLinkedDeque<T>();
      recycling.put(length, bin);
    }
    StackCounter stackCounter = getRecycle_submit(length);
    if (null != stackCounter) stackCounter.increment(length);
    if (bin.size() < Math.max(1, (int) (1e8 / length))) {
      synchronized (bin) {
        if (!bin.contains(data)) bin.add(data);
      }
    }
    else {
      stackCounter = getFrees(length);
      if (null != stackCounter) stackCounter.increment(length);
    }
  }
  
  /**
   * Obtain double [ ].
   *
   * @param length the length
   * @return the double [ ]
   */
  public T obtain(int length) {
    ConcurrentLinkedDeque<T> bin = recycling.get(length);
    StackCounter stackCounter = getRecycle_get(length);
    if (null != stackCounter) stackCounter.increment(length);
    if (null != bin) {
      T data = bin.poll();
      if (null != data) {
        reset(data);
        return data;
      }
    }
    try {
      stackCounter = getAllocations(length);
      if (null != stackCounter) stackCounter.increment(length);
      return create(length);
    } catch (java.lang.OutOfMemoryError e) {
      try {
        clear();
        System.gc();
        return create(length);
      } catch (java.lang.OutOfMemoryError e2) {
        throw new OutOfMemoryError("Could not allocate " + length + " bytes", e2);
      }
    }
  }
  
  /**
   * Create t.
   *
   * @param length the length
   * @return the t
   */
  public abstract T create(int length);
  
  /**
   * Reset.
   *
   * @param data the data
   */
  public abstract void reset(T data);
  
  /**
   * Clear.
   */
  public void clear() {
    recycling.clear();
  }
  
  /**
   * Copy of double [ ].
   *
   * @param original the original
   * @return the double [ ]
   */
  public T copyOf(T original) {
    if (null == original) return null;
    T copy = obtain(length(original));
    System.arraycopy(original, 0, copy, 0, length(original));
    return copy;
  }
  
  /**
   * Gets allocations.
   *
   * @param length the length
   * @return the allocations
   */
  public StackCounter getAllocations(int length) {
    if (!isProfiling(length)) return null;
    return allocations;
  }
  
  /**
   * Gets frees.
   *
   * @param length the length
   * @return the frees
   */
  public StackCounter getFrees(int length) {
    if (!isProfiling(length)) return null;
    return frees;
  }
  
  /**
   * Gets recycle submit.
   *
   * @param length the length
   * @return the recycle submit
   */
  public StackCounter getRecycle_submit(int length) {
    if (!isProfiling(length)) return null;
    return recycle_submit;
  }
  
  /**
   * Gets recycle get.
   *
   * @param length the length
   * @return the recycle get
   */
  public StackCounter getRecycle_get(int length) {
    if (!isProfiling(length)) return null;
    return recycle_get;
  }
  
  /**
   * Is profiling boolean.
   *
   * @param length the length
   * @return the boolean
   */
  public boolean isProfiling(int length) {
    return length > profilingThreshold;
  }
  
  /**
   * Sets profiling.
   *
   * @param threshold the threshold
   * @return the profiling
   */
  public RecycleBin<T> setProfiling(int threshold) {
    this.profilingThreshold = threshold;
    return this;
  }
}
