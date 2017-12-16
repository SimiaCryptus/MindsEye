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
 */
public abstract class RecycleBin<T> {
  
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
  private static final ScheduledExecutorService garbageTruck = Executors.newScheduledThreadPool(1);
  
  static {
    if (RecycleBin.class.desiredAssertionStatus()) {
      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        DOUBLES.printProfiling(System.err);
      }));
    }
  }
  
  private final ConcurrentHashMap<Integer, ConcurrentLinkedDeque<T>> recycling = new ConcurrentHashMap<>();
  private final StackCounter allocations = new StackCounter();
  private final StackCounter recycle_submit = new StackCounter();
  private final StackCounter recycle_get = new StackCounter();
  private final StackCounter frees = new StackCounter();
  private int profilingThreshold = 32 * 1024;
  
  private RecycleBin() {
    super();
  }
  
  private void DoubleArrays() {
    garbageTruck.scheduleAtFixedRate(new Runnable() {
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
    }, 0, 10, TimeUnit.SECONDS);
  }
  
  protected abstract int length(T data);
  
  public void printProfiling(PrintStream out) {
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
   * Recycle.
   *
   * @param data the data
   */
  public void recycle(T data) {
    if (null == data) return;
    //if(null != data) return;
    int length = length(data);
    if (length < 256) return;
    ConcurrentLinkedDeque<T> bin = recycling.get(length);
    if (null == bin) {
      //System.err.println("New Recycle Bin: " + data.length);
      bin = new ConcurrentLinkedDeque<T>();
      recycling.put(length, bin);
    }
    if (bin.size() < Math.max(1, (int) (1e8 / length))) {
      StackCounter stackCounter = getRecycle_submit(length);
      if (null != stackCounter) stackCounter.increment(length);
      synchronized (bin) {
        if (!bin.contains(data)) bin.add(data);
      }
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
    if (null != bin) {
      T data = bin.poll();
      if (null != data) {
        StackCounter stackCounter = getRecycle_get(length);
        if (null != stackCounter) stackCounter.increment(length(data));
        reset(data);
        return data;
      }
    }
    try {
      StackCounter stackCounter = getAllocations(length);
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
  
  public abstract T create(int length);
  
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
  
  public StackCounter getAllocations(int length) {
    if (!isProfiling(length)) return null;
    return allocations;
  }
  
  public StackCounter getFrees(int length) {
    if (!isProfiling(length)) return null;
    return frees;
  }
  
  public StackCounter getRecycle_submit(int length) {
    if (!isProfiling(length)) return null;
    return recycle_submit;
  }
  
  public StackCounter getRecycle_get(int length) {
    if (!isProfiling(length)) return null;
    return recycle_get;
  }
  
  public boolean isProfiling(int length) {
    return length > profilingThreshold;
  }
  
  public RecycleBin<T> setProfiling(int threshold) {
    this.profilingThreshold = threshold;
    return this;
  }
}
