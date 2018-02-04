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

import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Singleton;

import java.util.UUID;
import java.util.function.Supplier;

/**
 * A base class for a network node providing cached lazy evaluation; It gaurantees a node is only evaluated once, and
 * only if and when needed.
 */
@SuppressWarnings("serial")
abstract class LazyResult extends ReferenceCountingBase implements DAGNode {
  
  /**
   * The Id.
   */
  public final UUID id;
  
  
  /**
   * Instantiates a new Lazy result.
   */
  public LazyResult() {
    this(UUID.randomUUID());
  }
  
  /**
   * Instantiates a new Lazy result.
   *
   * @param id the id
   */
  protected LazyResult(final UUID id) {
    super();
    this.id = id;
  }
  
  /**
   * Eval nn result.
   *
   * @param t the t
   * @return the nn result
   */
  protected abstract NNResult eval(GraphEvaluationContext t);
  
  @Override
  public CountingNNResult get(final GraphEvaluationContext context) {
    long expectedCount = context.expectedCounts.getOrDefault(id, -1L);
    if (!context.calculated.containsKey(id)) {
      Singleton singleton = null;
      synchronized (context) {
        if (!context.calculated.containsKey(id)) {
          singleton = new Singleton();
          context.calculated.put(id, singleton);
        }
      }
      if (null != singleton) {
        NNResult result = eval(context);
        if (null == result) throw new IllegalStateException();
        singleton.set(new CountingNNResult(result));
        result.freeRef();
      }
    }
    Supplier<CountingNNResult> resultSupplier = context.calculated.get(id);
    if (null == resultSupplier) throw new IllegalStateException();
    CountingNNResult nnResult = null == resultSupplier ? null : resultSupplier.get();
    if (null == nnResult) throw new IllegalStateException();
    int references = nnResult.getAccumulator().increment();
    if (references <= 0) throw new IllegalStateException();
    if (expectedCount >= 0 && references > expectedCount) throw new IllegalStateException();
    nnResult.addRef();
    nnResult.getData().addRef();
    return nnResult;
  }
  
  @Override
  public final UUID getId() {
    return id;
  }
  
}
