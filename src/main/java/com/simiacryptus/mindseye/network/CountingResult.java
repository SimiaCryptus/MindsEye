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

import com.simiacryptus.mindseye.lang.CoreSettings;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.LinkedList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * A result type for evaluating the backpropigation phase of an Acyclic Directed Graph. Since the result of a given
 * compoent in a network can be used multiple times, we can improve efficiency of backprop by accumulating all the
 * backpropigating delta signals into a single signal before evaluating further backwards.
 */
public class CountingResult extends Result {
  /**
   * The constant logger.
   */
  protected static final Logger logger = LoggerFactory.getLogger(CountingResult.class);
  
  /**
   * The Inner.
   */
  @Nonnull
  private final Result inner;
  
  /**
   * Instantiates a new Counting nn result.
   *
   * @param inner the heapCopy
   */
  public CountingResult(@Nonnull final Result inner) {
    super(inner.getData(), new CountingAccumulator(inner));
    this.inner = inner;
    inner.addRef();
  }
  
  /**
   * Instantiates a new Counting result.
   *
   * @param r       the r
   * @param samples the samples
   */
  public CountingResult(final Result r, final int samples) {
    this(r);
    getAccumulator().references.set(samples);
  }
  
  @Nonnull
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
  static class CountingAccumulator extends ReferenceCountingBase implements BiConsumer<DeltaSet<Layer>, TensorList> {
    @Nonnull
    private final AtomicInteger references;
    @Nonnull
    private final AtomicBoolean hasAccumulated;
    private final Result inner;
    @Nonnull
    private final LinkedList<TensorList> passbackBuffers;
    @Nonnull
    private final AtomicInteger accumulations;
  
    /**
     * Instantiates a new Counting accumulator.
     *
     * @param inner the inner
     */
    public CountingAccumulator(Result inner) {
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
  
    /**
     * Gets count.
     *
     * @return the count
     */
    public int getCount() {
      return this.references.get();
    }
  
    @Override
    public void accept(DeltaSet<Layer> buffer, @Nonnull TensorList data) {
      //assert null == CudaSystem.getThreadHandle();
      assertAlive();
      data.assertAlive();
      if (1 >= references.get()) {
        data.addRef();
        inner.accumulate(buffer, data);
      }
      else {
        @Nonnull TensorList reduced = null;
        synchronized (passbackBuffers) {
          assert passbackBuffers.stream().allMatch(x -> x.assertAlive());
          passbackBuffers.add(data);
          data.addRef();
          if (passbackBuffers.size() > CoreSettings.INSTANCE.backpropAggregationSize) {
            Stream<TensorList> stream = passbackBuffers.stream();
            if (!CoreSettings.INSTANCE.isSingleThreaded()) stream = stream.parallel();
            //x.addRef();
            @Nonnull TensorList compacted = stream.reduce((a, b) -> {
              TensorList c;
              c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get();
            //passbackBuffers.stream().distinct().filter((TensorList x) -> x != reduced).forEach(t -> t.freeRef());
            passbackBuffers.clear();
            passbackBuffers.add(compacted);
            assert passbackBuffers.stream().allMatch(x -> x.assertAlive());
          }
          if (accumulations.incrementAndGet() == references.get()) {
            Stream<TensorList> stream = passbackBuffers.stream();
            if (!CoreSettings.INSTANCE.isSingleThreaded()) stream = stream.parallel();
            reduced = stream.reduce((a, b) -> {
              TensorList c;
              c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get();
            passbackBuffers.clear();
          }
          assert passbackBuffers.stream().allMatch(x -> x.assertAlive());
        }
        if (null != reduced) {
          inner.accumulate(buffer, reduced);
          accumulations.set(0);
        }
      }
    }
  
    @Override
    protected void _free() {
      synchronized (passbackBuffers) {
        passbackBuffers.stream().forEach(t -> t.freeRef());
        passbackBuffers.clear();
      }
      this.inner.freeRef();
    }
    
    
  }
}
