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

import com.simiacryptus.mindseye.lang.ReferenceCountingBase;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * This class provides the index for re-using the output of any given node during a single network evaluation, such that
 * each node is executed minimally.
 */
class GraphEvaluationContext extends ReferenceCountingBase {
  
  /**
   * The Expected counts.
   */
  final Map<UUID, Long> expectedCounts = new ConcurrentHashMap<>();
  
  /**
   * The Calculated.
   */
  final Map<UUID, Supplier<CountingNNResult>> calculated = new ConcurrentHashMap<>();
  
  @Override
  protected synchronized void _free() {
    calculated.entrySet().stream().filter(e -> {
      Supplier<CountingNNResult> x = e.getValue();
      CountingNNResult countingNNResult = x.get();
      if (expectedCounts.containsKey(e.getKey())) {
        return expectedCounts.get(e.getKey()) < countingNNResult.getAccumulator().getCount();
      }
      else {
        return true;
      }
    }).forEach(x -> {
      CountingNNResult result = x.getValue().get();
      result.freeRef();
      result.getData().freeRef();
    });
    calculated.clear();
  }
}
