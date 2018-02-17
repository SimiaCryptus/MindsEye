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
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Serializable;
import java.util.UUID;

/**
 * This is a logical node used within a network graph definition. It is associated with a LayerBase WITHOUT a 1:1
 * relation assumtion; i.e. the same logical layer CAN be used twice in the same graph. Also, the layer assigned to a
 * node can be updated, which can be useful for adding/removing instrumentation wrappers.
 */
public interface DAGNode extends Serializable, ReferenceCounting {
  
  /**
   * Get nn result.
   *
   * @param buildExeCtx the build exe ctx
   * @return the nn result
   */
  @javax.annotation.Nullable
  NNResult get(GraphEvaluationContext buildExeCtx);
  
  /**
   * Gets id.
   *
   * @return the id
   */
  UUID getId();
  
  /**
   * Get inputs dag node [ ].
   *
   * @return the dag node [ ]
   */
  @Nonnull
  default DAGNode[] getInputs() {
    return new DAGNode[]{};
  }
  
  /**
   * Gets layer.
   *
   * @param <T> the type parameter
   * @return the layer
   */
  @Nullable
  <T extends Layer> T getLayer();
  
  
  /**
   * Sets layer.
   *
   * @param layer the layer
   */
  void setLayer(Layer layer);
  
  /**
   * Gets network.
   *
   * @return the network
   */
  DAGNetwork getNetwork();
  
}