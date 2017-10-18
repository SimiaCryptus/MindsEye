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

package com.simiacryptus.mindseye.layers.meta;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SqActivationLayer;
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;

/**
 * The type Std dev meta layer.
 */
@SuppressWarnings("serial")
public class AvgNormalizationMetaLayer extends DAGNetwork {
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @return the nn layer
   */
  public static NNLayer fromJson(JsonObject inner) {
    return new AvgNormalizationMetaLayer(inner);
  }

  /**
   * Instantiates a new Std dev meta layer.
   *
   * @param json the json
   */
  protected AvgNormalizationMetaLayer(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgNormalizationMetaLayer.class);
  private final DAGNode head;
  
  /**
   * Instantiates a new Std dev meta layer.
   */
  public AvgNormalizationMetaLayer() {
    super(1);
    DAGNode centered = add(new SumInputsLayer(),
      getInput(0),
      add(new LinearActivationLayer().setScale(-1), add(new AvgReducerLayer(), getInput(0)))
    );
    this.head = add(new ProductInputsLayer(),
      centered,
      add(new NthPowerActivationLayer().setPower(-0.5), add(new AvgReducerLayer(), add(new SqActivationLayer(), centered)))
    );
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
}
