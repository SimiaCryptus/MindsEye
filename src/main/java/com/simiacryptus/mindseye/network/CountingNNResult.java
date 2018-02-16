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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.test.TestUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
  @javax.annotation.Nonnull
  private final NNResult inner;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner the heapCopy
   */
  protected CountingNNResult(@javax.annotation.Nonnull final NNResult inner) {
    super(inner.getData(), new CountingAccumulator(inner));
    this.inner = inner;
    inner.addRef();
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
  
  @javax.annotation.Nonnull
  @Override
  public CountingAccumulator getAccumulator() {
    return (CountingAccumulator) super.getAccumulator();
  }
  
  @Override
  protected void _free() {
    inner.freeRef();
    ((CountingAccumulator) accumulator).freeRef();
  }
  
  @Override
  public boolean isAlive() {
    return inner.isAlive();
  }
  
  /**
   * The type Counting accumulator.
   */
  static class CountingAccumulator extends ReferenceCountingBase implements BiConsumer<DeltaSet<NNLayer>, TensorList> {
    @javax.annotation.Nonnull
    private final AtomicInteger references;
    @javax.annotation.Nonnull
    private final AtomicBoolean hasAccumulated;
    private final NNResult inner;
    @javax.annotation.Nonnull
    private final LinkedList<TensorList> passbackBuffers;
    @javax.annotation.Nonnull
    private final AtomicInteger accumulations;
  
    /**
     * Instantiates a new Counting accumulator.
     *
     * @param inner the inner
     */
    public CountingAccumulator(NNResult inner) {
      this.inner = inner;
      this.inner.addRef();
      references = new AtomicInteger(0);
      hasAccumulated = new AtomicBoolean(false);
      passbackBuffers = new LinkedList<>();
      accumulations = new AtomicInteger(0);
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
    public void accept(DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull TensorList data) {
      assertAlive();
      if (1 >= references.get()) {
        if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
        inner.accumulate(buffer, data);
      }
      else {
        synchronized (passbackBuffers) {
          passbackBuffers.add(data);
          data.addRef();
          if (passbackBuffers.size() > COMPACTION_SIZE) {
            Stream<TensorList> stream = passbackBuffers.stream();
            if (!TestUtil.CONSERVATIVE) stream = stream.parallel();
            //x.addRef();
            @javax.annotation.Nonnull TensorList reduced = stream.reduce((a, b) -> {
              TensorList c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get();
            //passbackBuffers.stream().distinct().filter((TensorList x) -> x != reduced).forEach(t -> t.freeRef());
            passbackBuffers.clear();
            passbackBuffers.add(reduced);
          }
          if (accumulations.incrementAndGet() == references.get()) {
            if (hasAccumulated.getAndSet(true)) throw new IllegalStateException();
            //stream0 = stream0.parallel();
            //x.addRef();
            Stream<TensorList> stream = passbackBuffers.stream();
            if (!TestUtil.CONSERVATIVE) stream = stream.parallel();
            @javax.annotation.Nonnull TensorList reduced = stream.reduce((a, b) -> {
              TensorList c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get();
            inner.accumulate(buffer, reduced);
            reduced.freeRef();
            accumulations.set(0);
            passbackBuffers.clear();
          }
        }
      }
    }
  
    @Override
    protected void _free() {
      synchronized (passbackBuffers) {
        passbackBuffers.stream().distinct().forEach(t -> t.freeRef());
        passbackBuffers.clear();
      }
      this.inner.freeRef();
    }
    
    
  }
}
