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
import java.util.Deque;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
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
  private static final int COMPACTION_SIZE = 4;
  
  /**
   * The constant debugLifecycle.
   */
  public static boolean debugLifecycle = true;
  
  /**
   * The Inner.
   */
  private final NNResult inner;
  private final long expectedCount;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner         the heapCopy
   * @param expectedCount the expected count
   */
  protected CountingNNResult(final NNResult inner, long expectedCount) {
    super(inner.getData(), new CountingAccumulator(inner));
    this.inner = inner;
    this.expectedCount = expectedCount;
  }
  
  @Override
  public CountingAccumulator getAccumulator() {
    return (CountingAccumulator) super.getAccumulator();
  }
  
  /**
   * Mini stack trace string.
   *
   * @return the string
   */
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
    getAccumulator()._free();
    inner.free();
  }
  
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
  
  /**
   * The type Counting accumulator.
   */
  static class CountingAccumulator implements BiConsumer<DeltaSet<NNLayer>, TensorList> {
    private final AtomicInteger finalizations;
    private final AtomicInteger references;
    private final AtomicBoolean hasAccumulated;
    private final AtomicBoolean hasFinalized;
    private final NNResult inner;
    private final Deque<TensorList> passbackBuffers;
    /**
     * The Queued.
     */
    private final AtomicInteger accumulations;
    /**
     * The Finalized by.
     */
    public StackTraceElement[] finalizedBy;
    
    /**
     * Instantiates a new Counting accumulator.
     *
     * @param inner the inner
     */
    public CountingAccumulator(NNResult inner) {
      this.inner = inner;
      finalizations = new AtomicInteger(0);
      references = new AtomicInteger(0);
      hasAccumulated = new AtomicBoolean(false);
      hasFinalized = new AtomicBoolean(false);
      finalizedBy = null;
      passbackBuffers = new LinkedBlockingDeque<>();
      accumulations = new AtomicInteger(0);
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
    public int increment() {
      return this.references.incrementAndGet();
    }
    
    @Override
    public void accept(DeltaSet<NNLayer> buffer, TensorList data) {
      if (hasFinalized.get()) throw new IllegalStateException(finalizedByStr());
      if (1 >= references.get()) {
        if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
        inner.accumulate(buffer, data);
      }
      else {
        passbackBuffers.add(data);
        data.addRef();
        synchronized (passbackBuffers) {
          if (passbackBuffers.size() > COMPACTION_SIZE) {
            TensorList reduced = passbackBuffers.stream().parallel().reduce((a, b) -> a.add(b)).get();
            passbackBuffers.stream().distinct().filter((TensorList x) -> x != reduced).forEach(t -> t.freeRef());
            passbackBuffers.clear();
            passbackBuffers.add(reduced);
          }
          if (accumulations.incrementAndGet() == references.get()) {
            if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
            TensorList reduced = passbackBuffers.stream().parallel().reduce((a, b) -> a.add(b)).get();
            passbackBuffers.stream().distinct().filter((TensorList x) -> x != reduced).forEach(t -> t.freeRef());
            inner.accumulate(buffer, reduced);
            reduced.freeRef();
            accumulations.set(0);
            passbackBuffers.clear();
          }
        }
      }
    }
    
    /**
     * A flagrant abuse of Java's object finalization contract. Repeated calls to this class's free method will
     * increment a counter, and when the counter cycles the call is chained.
     */
    protected void _free() {
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
          synchronized (passbackBuffers) {
            passbackBuffers.stream().distinct().forEach(t -> t.freeRef());
            passbackBuffers.clear();
          }
        }
      }
    }
    
    
  }
}
