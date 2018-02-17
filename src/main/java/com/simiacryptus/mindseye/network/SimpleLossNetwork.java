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

import com.simiacryptus.mindseye.lang.Layer;

import javax.annotation.Nullable;

/**
 * The basic supervised network archetype. The network has two inputs; the input and the examplar output. A given
 * component is then evaluated on the input, and the resulting output is combined with the exemplar output via the loss
 * function.
 */
@SuppressWarnings("serial")
public class SimpleLossNetwork extends SupervisedNetwork {
  
  
  /**
   * The Loss node.
   */
  @Nullable
  public final DAGNode lossNode;
  /**
   * The Student node.
   */
  @Nullable
  public final DAGNode studentNode;
  
  /**
   * Instantiates a new Simple loss network.
   *
   * @param student the student
   * @param loss    the loss
   */
  public SimpleLossNetwork(@javax.annotation.Nonnull final Layer student, @javax.annotation.Nonnull final Layer loss) {
    super(2);
    studentNode = add(student, getInput(0));
    lossNode = add(loss, studentNode, getInput(1));
  }
  
  @Override
  public DAGNode getHead() {
    return lossNode;
  }
}
