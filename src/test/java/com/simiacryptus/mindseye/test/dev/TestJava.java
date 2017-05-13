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

package com.simiacryptus.mindseye.test.dev;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

public class TestJava {
  
  private static final Logger log = LoggerFactory.getLogger(TestJava.class);
  
  private void test(final Runnable task) {
    test(task, Thread.currentThread().getStackTrace()[2].toString());
  }
  
  private void test(final Runnable task, final String label) {
    final long start = System.nanoTime();
    task.run();
    final double duration = (System.nanoTime() - start) * 1.0e-9;
    log.debug(String.format("Tested %s in %.05fs", label, duration));
  }
  
  @Test
  public void testTightStreamLoop() throws Exception {
    {
      final int[] x = new int[1000000];
      test(() -> {
        for (int i = 0; i < x.length; i++) {
          x[i] += 1;
        }
      });
    }
    {
      final int[] x = new int[1000000];
      test(() -> {
        IntStream.range(0, x.length).forEach(i -> x[i] += 1);
      });
    }
    final int[] x = new int[1000000];
    test(() -> {
      IntStream.range(0, x.length).parallel().forEach(i -> x[i] += 1);
    });
  }
  
}
