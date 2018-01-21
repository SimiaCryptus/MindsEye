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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * A result type for evaluating the backpropigation phase of an Acyclic Directed Graph. Since the result of a given
 * compoent in a network can be used multiple times, we can improve efficiency of backprop by accumulating all the
 * backpropigating delta signals into a single signal before evaluating further backwards.
 */
class CountingNNResult extends NNResult {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CountingNNResult.class);
  
  /**
   * The constant debugLifecycle.
   */
  public static boolean debugLifecycle = true;
  
  /**
   * The Inner.
   */
  private final NNResult inner;
  /**
   * The Queued.
   */
  private final AtomicInteger accumulations = new AtomicInteger(0);
  /**
   * The Finalizations.
   */
  private final AtomicInteger finalizations = new AtomicInteger(0);
  private final AtomicInteger references = new AtomicInteger(0);
  private final AtomicBoolean hasAccumulated = new AtomicBoolean(false);
  private final AtomicBoolean hasFinalized = new AtomicBoolean(false);
  /**
   * The Finalized by.
   */
  public StackTraceElement[] finalizedBy = null;
  /**
   * The Passback buffer.
   */
  volatile TensorList passbackBuffer = null;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner the heapCopy
   */
  protected CountingNNResult(final NNResult inner) {
    super(inner.getData());
    this.inner = inner;
    logger.info(String.format("%s.<init>(%s) via %s", this, inner, miniStackTrace()));
  }
  
  public static String miniStackTrace() {
    int max = 30;
    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    List<String> list = Arrays.stream(stackTrace).skip(3).limit(max - 3).map(x -> x.isNativeMethod() ? "(Native Method)" :
      (x.getFileName() != null && x.getLineNumber() >= 0 ?
        x.getFileName() + ":" + x.getLineNumber() :
        (x.getFileName() != null ? x.getFileName() : "(Unknown Source)"))).collect(Collectors.toList());
    return "[" + list.stream().reduce((a, b) -> a + ", " + b).get() + (stackTrace.length > max ? ", ..." : "") + "]";
  }
  
  /**
   * A flagrant abuse of Java's object finalization contract. Repeated calls to this class's free method will increment
   * a counter, and when the counter cycles the call is chained.
   */
  @Override
  public void free() {
    logger.info(String.format("%s.free(%s/%s)", this, finalizations.get(), references.get()));
    if (1 >= references.get()) {
      if (!hasFinalized.getAndSet(true)) {
        finalizedBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
        inner.free();
      }
    }
    else {
      if (finalizations.incrementAndGet() == references.get()) {
        if (!hasFinalized.getAndSet(true)) {
          finalizedBy = debugLifecycle ? Thread.currentThread().getStackTrace() : null;
          inner.free();
        }
        finalizations.set(0);
        passbackBuffer = null;
      }
    }
  }
  
  @Override
  public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
    logger.info(String.format("%s.accumulate(%s/%s,%s) via %s", this, accumulations.get(), references.get(), data, miniStackTrace()));
    if (hasFinalized.get()) throw new IllegalStateException(finalizedByStr());
    if (1 >= references.get()) {
      if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
      inner.accumulate(buffer, data);
    }
    else {
      boolean created = false;
      if (null == passbackBuffer) {
        synchronized (this) {
          if (null == passbackBuffer) {
            passbackBuffer = data.copy();
            created = true;
          }
        }
      }
      if (null != passbackBuffer && !created) {
        passbackBuffer.addInPlace(data);
      }
      if (accumulations.incrementAndGet() == references.get()) {
        if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
        inner.accumulate(buffer, passbackBuffer);
        accumulations.set(0);
        passbackBuffer = null;
      }
    }
  }
  
  /**
   * Finalized by str string.
   *
   * @return the string
   */
  public String finalizedByStr() {
    return null == finalizedBy ? "" : Arrays.stream(finalizedBy).map(x -> x.toString()).reduce((a, b) -> a + "; " + b).get();
  }
  
  /**
   * Increment counting nn result.
   *
   * @return the counting nn result
   */
  public CountingNNResult increment() {
    logger.info(String.format("%s.increment(%s/%s) via %s", this, accumulations.get(), references.get(), miniStackTrace()));
    this.references.incrementAndGet();
    return this;
  }
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
}
