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

import com.simiacryptus.util.data.DoubleStatistics;

import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class StackCounter {
  
  ConcurrentHashMap<StackTraceElement, DoubleStatistics> stats = new ConcurrentHashMap<>();
  
  public void increment(int length) {
    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    for (StackTraceElement frame : stackTrace) {
      stats.computeIfAbsent(frame, f -> new DoubleStatistics()).accept(length);
    }
  }
  
  @Override
  public String toString() {
    Comparator<Map.Entry<StackTraceElement, DoubleStatistics>> comparing = Comparator.comparing(e -> -summaryStat(e.getValue()).doubleValue());
    comparing = comparing.thenComparing(Comparator.comparing(e -> e.getKey().toString()));
    return stats.entrySet().stream()
      .sorted(comparing)
      .map(e -> String.format("%s - %s", toString(e.getKey()), summaryStat(e.getValue())))
      .limit(100).reduce((a, b) -> a + "\n" + b).orElse(super.toString());
  }
  
  protected String toString(StackTraceElement frame) {
    return String.format(
      "%s.%s(%s:%s)",
      frame.getClassName(),
      frame.getMethodName(),
      frame.getFileName(),
      frame.getLineNumber()
    );
  }
  
  protected Number summaryStat(DoubleStatistics value) {
    return (int) value.getSum();
  }
}
