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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.ComponentException;
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNResult;

import java.util.UUID;

/**
 * A base class for a network node providing cached lazy evaluation; It gaurantees a node is only evaluated once, and
 * only if and when needed.
 */
@SuppressWarnings("serial")
abstract class LazyResult implements DAGNode {
  
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
   * @param t         the t
   * @param nncontext the nncontext
   * @return the nn result
   */
  protected abstract NNResult eval(GraphEvaluationContext t, NNExecutionContext nncontext);
  
  @Override
  public synchronized CountingNNResult get(final NNExecutionContext nncontext, final GraphEvaluationContext t) {
    return t.cache.computeIfAbsent(id, k -> {
      try {
        return new CountingNNResult(eval(t, nncontext));
      } catch (final Throwable e) {
        throw new ComponentException("Error evaluating layer " + getLayer(), e);
      }
    }).increment();
  }
  
  @Override
  public final UUID getId() {
    return id;
  }
  
}
