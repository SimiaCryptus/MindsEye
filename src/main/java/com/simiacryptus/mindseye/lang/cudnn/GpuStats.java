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

package com.simiacryptus.mindseye.lang.cudnn;

import java.util.concurrent.atomic.AtomicLong;

/**
 * The type Gpu stats.
 */
public class GpuStats {
  /**
   * The Memory reads.
   */
  public final AtomicLong memoryReads = new AtomicLong(0);
  /**
   * The Memory writes.
   */
  public final AtomicLong memoryWrites = new AtomicLong(0);
  /**
   * The Peak memory.
   */
  public final AtomicLong peakMemory = new AtomicLong(0);
  /**
   * The Used memory.
   */
  public final AtomicLong usedMemory = new AtomicLong(0);
  
  public final int highMemoryThreshold = 6 * 1024 * 1024 * 1024;
}
