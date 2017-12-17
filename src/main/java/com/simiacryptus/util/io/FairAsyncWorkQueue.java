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

package com.simiacryptus.util.io;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The type Fair async work queue.
 */
public class FairAsyncWorkQueue implements Runnable {
  private final AtomicBoolean isRunning = new AtomicBoolean(false);
  private final ExecutorService pool = Executors.newCachedThreadPool();
  private final LinkedBlockingDeque<Runnable> queue = new LinkedBlockingDeque<>();
  
  @Override
  public void run() {
    if (isRunning.getAndSet(true)) {
      try {
        while (true) {
          final Runnable poll = queue.poll();
          if (null != poll) {
            poll.run();
          }
          else {
            break;
          }
        }
      } finally {
        isRunning.set(false);
      }
    }
  }
  
  /**
   * Submit.
   *
   * @param task the task
   */
  public void submit(final Runnable task) {
    queue.add(task);
    pool.submit(this);
  }
}
