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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SqActivationLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;

/**
 * The type Std dev meta layer.
 */
@SuppressWarnings("serial")
public class StdDevMetaLayer extends DAGNetwork {
  
  public static NNLayer fromJson(JsonObject inner) {
    return new StdDevMetaLayer(inner);
  }

  protected StdDevMetaLayer(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StdDevMetaLayer.class);
  private final DAGNode head;
  
  /**
   * Instantiates a new Std dev meta layer.
   */
  public StdDevMetaLayer() {
    super(1);
    this.head = add(new NthPowerActivationLayer().setPower(0.5),
      add(new SumInputsLayer(),
        add(new AvgMetaLayer(), add(new SqActivationLayer(), getInput(0))),
        add(new LinearActivationLayer().setScale(-1), add(new SqActivationLayer(), add(new AvgMetaLayer(), getInput(0))))
      ));
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
}
