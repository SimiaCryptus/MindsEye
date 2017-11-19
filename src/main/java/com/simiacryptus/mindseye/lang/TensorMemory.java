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

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class TensorMemory {
  
  static {
    java.util.concurrent.Executors.newScheduledThreadPool(1).scheduleAtFixedRate(new Runnable() {
      @Override
      public void run() {
        recycling.forEach((k, v) -> {
          v.clear();
        });
      }
    }, 0, 10, TimeUnit.SECONDS);
  }
  
  private static ConcurrentHashMap<Integer, BlockingQueue<double[]>> recycling = new ConcurrentHashMap<>();
  
  private TensorMemory() {
    super();
  }
  
  /**
   * Recycle.
   *
   * @param data the data
   */
  public static void recycle(double[] data) {
    if (null == data) return;
    //if(null != data) return;
    if (data.length < 256) return;
    BlockingQueue<double[]> bin = recycling.get(data.length);
    if (null == bin) {
      //System.err.println("New Recycle Bin: " + data.length);
      bin = new ArrayBlockingQueue<double[]>(Math.max(1,(int) (1e8 / data.length)));
      recycling.put(data.length, bin);
    }
    bin.offer(data);
  }
  
  /**
   * Obtain double [ ].
   *
   * @param length the length
   * @return the double [ ]
   */
  public static double[] obtain(int length) {
    BlockingQueue<double[]> bin = recycling.get(length);
    if (null != bin) {
      double[] data = bin.poll();
      if (null != data) {
        Arrays.fill(data, 0);
        return data;
      }
    }
    try {
      return new double[length];
    } catch (java.lang.OutOfMemoryError e) {
      try {
        clear();
        System.gc();
        return new double[length];
      } catch (java.lang.OutOfMemoryError e2) {
        throw new OutOfMemoryError("Could not allocate " + length + " bytes", e2);
      }
    }
  }
  
  /**
   * Clear.
   */
  public static void clear() {
    recycling.clear();
  }
}
