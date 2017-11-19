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

import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;

import java.io.Serializable;
import java.util.UUID;

/**
 * The interface Dag node.
 */
public interface DAGNode extends Serializable {
  
  
  /**
   * Gets id.
   *
   * @return the id
   */
  UUID getId();
  
  /**
   * Gets layer.
   *
   * @param <T> the type parameter
   * @return the layer
   */
  <T extends NNLayer> T getLayer();
  
  /**
   * Sets layer.
   *
   * @param layer the layer
   */
  void setLayer(NNLayer layer);
  
  /**
   * Get nn result.
   *
   * @param nncontext   the nncontext
   * @param buildExeCtx the build exe ctx
   * @return the nn result
   */
  NNResult get(NNExecutionContext nncontext, GraphEvaluationContext buildExeCtx);
  
  
  /**
   * Get inputs dag node [ ].
   *
   * @return the dag node [ ]
   */
  default DAGNode[] getInputs() {
    return new DAGNode[]{};
  }
  
  
}